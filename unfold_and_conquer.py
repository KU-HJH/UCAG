
# CUDA_VISIBLE_DEVICES=0
import json
from xml.dom import INDEX_SIZE_ERR

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from kornia.filters.gaussian import gaussian_blur2d
from tqdm import tqdm
from cam import *
from parsing import batch_normalize_map, parsing, single_normalize_map

__all__ = ['CausalMetric', 'auc']

import warnings
warnings.filterwarnings("ignore")

HW = 224 * 224

# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class CausalMetric(object):
    def __init__(self, model, mode, step, substrate_fn):
        """Create deletion/insertion metric instance.
        Args:
            model(nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model.eval().cuda()
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def evaluate(self, img, pred, mask, cls_idx=None, b=16, save_to=None):
        """Run metric on one image-saliency pair.
        Args:
            img (Tensor): normalized image tensor.
            mask (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step.
            save_to (str): directory to save every step plots to.
        Return:
            scores (nd.array): Array containing scores at every step.
        """
        if cls_idx is None:
            cls_idx = pred

        n_steps = (HW + self.step - 1) // self.step
        if self.mode == 'del':
            title = 'Deletion Curve'
            ylabel = 'Pixels deleted'
            start = img.clone()
            # start = img.repeat(b, 1, 1, 1)
            finish = self.substrate_fn(img)
        elif self.mode == 'ins':
            title = 'Insertion Curve'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img)
            finish = img.clone()
        scores = np.empty((b, n_steps + 1), dtype='float32')
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(mask.reshape(b, HW), axis=1), axis=-1)

        for i in range(n_steps + 1):
            logit = self.model(start.cuda())
            score = F.softmax(logit, dim=-1)[torch.arange(0, b), cls_idx].squeeze()
            for j in range(b):
                scores[j, i] = score[j]
            
            coords = []
            for j in range(b):
                coords = salient_order[j, self.step * i: self.step * (i + 1)]
                start.cpu().numpy().reshape(b, 3, HW)[j, :, coords] = \
                    finish.cpu().numpy().reshape(b, 3, HW)[j, :, coords]

        aucs = np.empty(b, dtype='float32')
        for i in range(b):
            aucs[i] = auc(scores[i].reshape(-1))
    
        return aucs

map_blur = lambda x: gaussian_blur2d(x, kernel_size=(5, 5), sigma=(3., 3.))

import time

def multi_scale_fusion(cam, input, target, args, image_size=(224, 224)):
    saliency_maps = ucag(cam, input, target, args=args, image_size=image_size)
    return saliency_maps

def ucag(cam, input, target, args=None, image_size=None):
    crop_size=args.crop_size
    n=args.num_units
    padding=args.padding
    with_score=args.score
    alpha=args.alpha
    dilation=args.dilation
    limit_batch_size = args.limit_batch_size
    stride = args.stride
    
    input_size = (input.size()[-2], input.size()[-1])
    start = time.time()
    # input_size = 224
    with torch.no_grad():
        cam_input = align_generator(input, n=n, input_size=input_size, crop_size=crop_size, alpha=alpha, dilation=dilation, padding=padding, stride=stride)
    new_maps, score = list(), list()

    # batch-wise version
    if torch.is_tensor(target):
        target = target.item()
    target = torch.tensor([target,] * cam_input.size(0), device=cam_input.device)
    if image_size is None:
        h, w = input_size
    else:
        h, w = image_size
    
    if with_score:
        if args.divide:
            act, grad, logit = cam(cam_input, class_idx=target)
            score = F.softmax(logit, dim=1)[torch.arange(0, cam_input.size(0)), target]
            saliency_map = torch.mul(act, grad)
            saliency_map = align_combiner(saliency_map, score, n=n, input_size=input_size, crop_size=crop_size, dilation=dilation, padding=padding, stride=stride, multi=args.multi)
            saliency_map = saliency_map.sum(dim=1, keepdim=True)
        else:
            if cam_input.size(0) >= limit_batch_size:
                max_iter = cam_input.size(0) // limit_batch_size
                for idx in range(max_iter + 1):
                    if idx == 0:
                        new_maps, logits = cam(cam_input[idx * limit_batch_size :(idx + 1) * limit_batch_size , ...], 
                                    class_idx=target[idx * limit_batch_size :(idx + 1) * limit_batch_size , ...], return_logit=True, wonorm=True)
                    elif idx != max_iter:
                        new_map, logit = cam(cam_input[idx * limit_batch_size :(idx + 1) * limit_batch_size , ...], 
                                                            class_idx=target[idx * limit_batch_size :(idx + 1) * limit_batch_size , ...], return_logit=True, wonorm=True)
                        new_maps = torch.cat([new_maps, new_map], dim=0)
                        logits = torch.cat([logits, logit], dim=0)
                        
                    elif cam_input.size(0) % limit_batch_size != 0:
                        new_map, logit = cam(cam_input[idx * limit_batch_size :idx * limit_batch_size + (cam_input.size(0) % limit_batch_size), ...], 
                                                            class_idx=target[idx * limit_batch_size :idx * limit_batch_size + (cam_input.size(0) % limit_batch_size), ...], return_logit=True, wonorm=True)
                        new_maps = torch.cat([new_maps, new_map], dim=0)
                        logits = torch.cat([logits, logit], dim=0)
            else:
                new_maps, logits = cam(cam_input, class_idx=target, return_logit=True, wonorm=True)
            with torch.no_grad():
                logits = F.softmax(logits, dim=1)
                score = logits[torch.arange(0, cam_input.size(0)), target]
                saliency_map = align_combiner(new_maps, score, n=n, input_size=input_size, crop_size=crop_size, dilation=dilation, padding=padding, stride=stride, multi=args.multi)
    else:
        if cam_input.size(0) >= limit_batch_size:
            max_iter = cam_input.size(0) // limit_batch_size
            for idx in range(max_iter + 1):
                if idx == 0:
                    new_maps = cam(cam_input[idx * limit_batch_size :(idx + 1) * limit_batch_size , ...], 
                                   class_idx=target[idx * limit_batch_size :(idx + 1) * limit_batch_size , ...], return_logit=False, wonorm=True)
                elif idx != max_iter:
                    new_maps = torch.cat([new_maps, cam(cam_input[idx * limit_batch_size :(idx + 1) * limit_batch_size , ...], 
                                                        class_idx=target[idx * limit_batch_size :(idx + 1) * limit_batch_size , ...], return_logit=False, wonorm=True)], dim=0)
                elif cam_input.size(0) % limit_batch_size != 0:
                    new_maps = torch.cat([new_maps, cam(cam_input[idx * limit_batch_size :idx * limit_batch_size + (cam_input.size(0) % limit_batch_size), ...], 
                                                        class_idx=target[idx * limit_batch_size :idx * limit_batch_size + (cam_input.size(0) % limit_batch_size), ...], return_logit=False, wonorm=True)], dim=0)
        else:
            new_maps = cam(cam_input, class_idx=target, return_logit=False, wonorm=True)
        
        with torch.no_grad():
            saliency_map = align_combiner(new_maps, None, n=n, input_size=input_size, crop_size=crop_size, dilation=dilation, padding=padding, stride=stride, multi=args.multi)
    
    # saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
    if not args.cam in ['rap', 'rsp', 'agf', 'integrad', 'lrp', 'lrp_ab', 'clrp', 'sglrp', 'rollout', 'full_lrp', 'transformer_attribution', 'lrp_last_layer', 'attn_last_layer', 'attn_gradcam']:
        saliency_map = batch_normalize_map(saliency_map, image_size=(h + 2 * padding, w + 2 * padding))[:, :, padding:h+padding, padding:w+padding]
        saliency_map = map_blur(saliency_map)
    return saliency_map

# Modified set canvas size to crop size
# Trial3
def align_generator(images, n=6, input_size=(224, 224), crop_size=176, alpha=2.6, padding=0, dilation=1, stride=None):
    # get batch_size=1 image
    h, w = input_size
    canvas = torch.zeros(n * n, images.size(1), h, w, device=images.device)
    
    h_crop_size = int(crop_size * (h / 224))
    w_crop_size = int(crop_size * (w / 224))
    h = h + 2 * padding
    w = w + 2 * padding
    
    images = F.pad(images, mode='constant', pad=(padding,)*4)
    
    h_unit = (h - (h_crop_size - 1) * dilation + 1) // (n - 1)
    w_unit = (w - (w_crop_size - 1) * dilation + 1) // (n - 1)
    
    canvas = torch.zeros(n * n, images.size(1), int(h_crop_size * alpha) , int(w_crop_size * alpha), device=images.device)
    try:
        for i in range(n):
            for j in range(n):
                input = images[:, :, i * h_unit: h_crop_size + i * h_unit, j * w_unit: w_crop_size + j * w_unit]
                canvas[i * n + j, ...] = F.interpolate(input, size=(int(h_crop_size * alpha) , int(w_crop_size * alpha)), mode='bicubic', align_corners=False)
    except Exception as e:
        print(e)
        print(i * h_unit, ':', h_crop_size + i * h_unit, j * w_unit, ':', w_crop_size + j * w_unit, images.shape, images[:, :, i * h_unit: h_crop_size + i * h_unit, j * w_unit: w_crop_size + j * w_unit].shape)
        import sys; sys.exit()
    # print(h_crop_size * 2, w_crop_size * 2, h, w)
    
    return canvas

def align_combiner(maps, score, n=6, input_size=(224, 224), crop_size=176, padding=0, dilation=1, stride=None, multi=False):
    # get batch_size=1 image
    h, w = input_size
    h_crop_size = int(crop_size / 224 * h)
    w_crop_size = int(crop_size / 224 * w)
    h = h + 2 * padding
    w = w + 2 * padding
    
    if stride is not None:
        if not isinstance(stride, list):
            h_unit = stride
            w_unit = stride
        else:
            h_unit = stride[0]
            w_unit = stride[1]
        h_unit *= int(h / 224)
        w_unit *= int(w / 224)
        
        hn = int((h - dilation * (h_crop_size - 1) - 1) / h_unit + 1)
        wn = int((w - dilation * (w_crop_size - 1) - 1) / w_unit + 1)
        
        maps = F.interpolate(maps, size=(h_crop_size, w_crop_size), mode='bilinear', align_corners=False)
        if score is None: score = torch.ones(maps.size(0), device=maps.device) 
        maps = maps * score.view(-1, 1, 1, 1)
        maps = maps.view(hn*wn, h_crop_size * w_crop_size, 1).transpose(0, 2)
        canvas = F.fold(maps, output_size=(224, 224), kernel_size=(h_crop_size, w_crop_size), stride=(h_unit, w_unit), padding=(padding, padding), dilation=dilation)
        div_canvas = F.fold(torch.ones(1, h_crop_size * w_crop_size, hn * wn, device=maps.device), output_size=(224, 224), kernel_size=(h_crop_size, w_crop_size), stride=(h_unit, w_unit), padding=(padding, padding), dilation=dilation)
        map = torch.div(canvas, div_canvas)#[:, :, p:input_size[0]+p, p:input_size[1]+p]
        
    else:
        h_unit = (h - (h_crop_size - 1) * dilation + 1) // (n - 1)
        w_unit = (w - (w_crop_size - 1) * dilation + 1) // (n - 1)
        canvas = torch.zeros(1, maps.size(1), h_crop_size + (n - 1) * h_unit, w_crop_size + (n - 1) * w_unit, device=maps.device)
        div_canvas = torch.ones_like(canvas)
        maps = F.interpolate(maps, size=(h_crop_size, w_crop_size), mode='bilinear', align_corners=False)
        if score is None: score = torch.ones(maps.size(0), device=maps.device)
        score = score.exp() 
        score /= score.max()
        for i in range(n):
            for j in range(n):
                canvas[0, :, i * h_unit: h_crop_size + i * h_unit, j * w_unit: w_crop_size + j * w_unit] += \
                    maps[i * n + j, ...] * score[i * n + j].view(1, 1, 1)
                div_canvas[:, :, i * h_unit: h_crop_size + i * h_unit, j * w_unit: w_crop_size + j * w_unit] += 1
        map = torch.div(canvas, div_canvas)
    return map

