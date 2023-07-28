# CUDA_VISIBLE_DEVICES=0
import json

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
from parsing import *
from cam import *
from unfold_and_conquer import ucag, multi_scale_fusion
import os

__all__ = ['CausalMetric', 'auc']

import warnings
warnings.filterwarnings("ignore")

HW = 224 * 224
img_label = json.load(open('./utils/resources/imagenet_class_index.json', 'r'))


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


# Sampler for pytorch loader. Given range r loader will only
# return dataset[r] instead of whole dataset.
class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)


def explain_all(data_loader, explainer):
    """Get saliency maps for all Images in val loader
    Args:
        data_loader: torch data loarder
        explainer: gradcam, etc.
    Return:
        Images: list, length: len(data_loader), element: torch tensor with shape of (1, 3, H, W)
        explanations: np.ndarrays, with shape of (len(data_loader, H, W)
    """
    global vgg
    explanations = []
    images = []
    for i, (img, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Explaining Images')):
        try:
            # cls_idx = vgg(img.cuda()).max(1)[-1].item()
            saliency_maps = explainer(img.cuda(), class_idx=None).data
            explanations.append(saliency_maps.cpu().numpy())
            images.append(img)
        except Exception as e:
            continue

    explanations = np.array(explanations)
    return images, explanations


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

    def evaluate(self, img, pred, mask, idx=None, cls_idx=None, b=16, save_to=None, orig_score=None):
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
        HW = img.size(2) * img.size(3)
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

        stops = [n_steps // (i + 1) for i in range(10)]
        for i in range(n_steps + 1):
            logit = self.model(start.cuda())
            score = F.softmax(logit, dim=-1)[torch.arange(0, b), cls_idx].squeeze()
            # score = torch.min(torch.cat([score, orig_score], dim=0), dim=0)[0]
            for j in range(b):
                scores[j, i] = score[j]
            
            if i in stops and save_to:
                for j in range(b):
                    refered_scores = scores[j]
                    plt.figure(figsize=(10, 5))
                    plt.subplot(121)
                    plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, refered_scores[i]))
                    plt.axis('off')
                    tensor_imshow(start[j])

                    plt.subplot(122)
                    plt.plot(np.arange(i + 1) / n_steps, refered_scores[:i + 1])
                    plt.xlim(-0.1, 1.1)
                    plt.ylim(0, 1.05)
                    plt.fill_between(np.arange(i + 1) / n_steps, 0, refered_scores[:i + 1], alpha=0.4)
                    plt.title(title)
                    plt.xlabel(ylabel)
                    plt.ylabel(img_label[str(cls_idx[j].item())][-1])
                    plt.savefig(os.path.join(save_to, f'{self.mode}', '{}_{}.jpg'.format(j + idx * b, n_steps // (i + 1))))
                    plt.close()
            
            coords = []
            for j in range(b):
                coords = salient_order[j, self.step * i: self.step * (i + 1)]
                start.cpu().numpy().reshape(b, 3, HW)[j, :, coords] = \
                    finish.cpu().numpy().reshape(b, 3, HW)[j, :, coords]

        aucs = np.empty(b, dtype='float32')
        for i in range(b):
            aucs[i] = auc(scores[i].reshape(-1))
    
        return aucs

def main():
    # hyper-parameters
    val_dir = './data/val'
    fix_random_seeds()
    args = parsing()
    batch_size = args.batch_size
    num_workers = args.workers
    
    if args.save:
        save_root = args.save

    # model = models.resnet50(pretrained=True)
    model, input_size = get_model(args.name)
    model.eval()
    model = model.cuda()
    def correct_inplace_relu(model):
        for child_name, child in model.named_children():
            if isinstance(child, torch.nn.ReLU):
                setattr(model, child_name, torch.nn.ReLU())
            else:
                correct_inplace_relu(child)
    correct_inplace_relu(model)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = DataLoader(
        datasets.ImageFolder(val_dir, transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize([input_size, input_size]),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
        # sampler=RangeSampler(sample_range)
    )

    # Function that blurs input image
    blur = lambda x: gaussian_blur2d(x, kernel_size=(51, 51), sigma=(50., 50.))

    # Evaluate a batch of explanations

    insertion = CausalMetric(model, 'ins', int(input_size * 8), substrate_fn=blur)
    deletion = CausalMetric(model, 'del', int(input_size * 8), substrate_fn=torch.zeros_like)

    # args.image_size = input_size
    save_path = None
    scores = {'del': [], 'ins': []}
    cam = get_cam(args.cam, model, args.target_layer, args=args)
    image_size = (args.image_size, args.image_size)

    for i, (images, target) in enumerate(tqdm(val_loader, desc='Explaining Images')):
        target = target.cuda()
        images = images.cuda()
        if args.ours:
            for b in range(batch_size):
                input = images[b].unsqueeze(0)
                if b == 0:
                    saliency_maps = multi_scale_fusion(cam, input, target[b], args=args, image_size=(input_size, input_size))
                else:
                    saliency_map = multi_scale_fusion(cam, input, target[b].item(), args=args, image_size=(input_size, input_size))
                    saliency_maps = torch.cat([saliency_maps, saliency_map], dim=0)
        else:
            if args.alpha != 1.:
                images_ = F.interpolate(images, size=(int(224 * args.alpha),) * 2, mode='bicubic', align_corners=False)
            else:
                images_ = images
            # else:
            saliency_maps, logit = cam(images_, class_idx=target, return_logit=True, image_size=(input_size, input_size))
        
        saliency_maps = saliency_maps.data.cpu().numpy()
        if args.save:
            save_path = os.path.join(save_root, f'{target[0]}')
            os.makedirs(os.path.join(save_path, 'ins'), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'del'), exist_ok=True)
        del_score = deletion.evaluate(img=images.detach().cpu(), pred=target, mask=saliency_maps, idx=i, b=batch_size, save_to=save_path)
        ins_score = insertion.evaluate(img=images.detach().cpu(), pred=target, mask=saliency_maps, idx=i, b=batch_size, save_to=save_path)

        if args.debug and (i == 100):
                break

        for delscore in del_score:
            scores['del'].append(delscore)
        for insscore in ins_score:
            scores['ins'].append(insscore)
    print('----------------------------------------------------------------')
    print('Final:\nDeletion - {:.5f}\nInsertion - {:.5f}'.format(np.mean(scores['del']), np.mean(scores['ins'])))

if __name__ == '__main__':
    main()