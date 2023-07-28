from re import I
import cv2
import torch
import torchvision.models as models
import matplotlib.cm as cm
from torchvision import transforms
import numpy as np
from data.imagenet_labelmap2 import get_labelmap
from cam import *
from demo_eval import ucag, multi_scale_fusion

import os
import argparse
from PIL import Image
from tqdm import tqdm
from ATTR import render
from parsing import batch_normalize_map, get_cam, get_model

normalizeTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalizeImageTransform = transforms.Compose([transforms.ToTensor(), normalizeTransform])

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('cam', type=str)
    parser.add_argument('-m', '--name', type=str, default='resnet50')
    parser.add_argument('--img-path', type=str, default='./data/val')
    parser.add_argument('--save-path', type=str, default='Results')
    parser.add_argument('-i', '--img-size', type=int, default=224)
    parser.add_argument('--score', action='store_true')
    parser.add_argument('--multi', action='store_true')
    parser.add_argument('--divide', action='store_true')
    parser.add_argument('--dir', action='store_true')
    parser.add_argument('-o', '--ours', action='store_true')
    parser.add_argument('--lrp-based', action='store_true')
    parser.add_argument('--is-ablation', action='store_true')
    parser.add_argument('-t', '--target-layer', default='layer4', type=str)
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('-c', '--crop-size', default=124, type=int)
    parser.add_argument('-a', '--alpha', default=2.6, type=float)
    parser.add_argument('-d', '--dilation', default=1, type=int)
    parser.add_argument('-p', '--padding', default=0, type=int)
    parser.add_argument('-l', '--limit-batch-size', default=16, type=int)
    parser.add_argument('--stride', default=None, type=int, nargs='+')
    parser.add_argument('-n', '--num-units', default=6, type=int)
    
# Running original gradcam --
# python demo_dir.py gradcam 
# Running original gradcam + UCAG
# python demo_dir.py gradcam -c 124 -n 6 -a 2.6 --ours --score
    
    
    # AGFVis
    parser.add_argument('--arc', type=str, default='vgg', metavar='N',
                        help='Model architecture')
    parser.add_argument('--train_dataset', type=str, default='imagenet', metavar='N',
                        help='Testing Dataset')
    parser.add_argument('--method', type=str,
                        default='agf',
                        help='')
    parser.add_argument('--thr', type=float, default=0.,
                        help='threshold')

    # Ablation option from the our method
    parser.add_argument('--no-a', action='store_true',
                        default=False,
                        help='No A')
    parser.add_argument('--no-fx', action='store_true',
                        default=False,
                        help='No F_x')
    parser.add_argument('--no-fdx', action='store_true',
                        default=False,
                        help='No F_dx')
    parser.add_argument('--no-m', action='store_true',
                        default=False,
                        help='No M')
    parser.add_argument('--no-reg', action='store_true',
                        default=False,
                        help='No regulatization by C')
    parser.add_argument('--gradcam', action='store_true',
                        default=False,
                        help='Use GradCAM method as residual')
    
    
    return parser.parse_args()
    

def loadImage(imagePath, imageSize):
    rawImage = cv2.imread(imagePath)
    # rawImage = cv2.resize(rawImage, (224,) * 2, interpolation=cv2.INTER_LINEAR)
    rawImage = cv2.resize(rawImage, (imageSize,) * 2, interpolation=cv2.INTER_LINEAR)
    image = normalizeImageTransform(rawImage[..., ::-1].copy())
    return image, rawImage

def saveMapWithColorMap(filename, map, image):
    cmap = cm.jet_r(map)[..., :3] * 255.0
    map = (cmap.astype(np.float) + image.astype(np.float)) / 2
    if not cv2.imwrite(filename, np.uint8(map)):
        print('sibar')

def get_image_list(args):
    label_map_file = open('data/imagenet_labelmap.txt', 'r')
    lines = label_map_file.readlines()
    key_map = dict()
    for line in lines:
        cls_str, _, cls_name = line.replace('\n', '').split(' ')
        key_map[cls_name] = cls_str
    original_labelmap = get_labelmap()
    holder = dict()
    for k, v in original_labelmap.items():
        holder[k] = v.split(',')[0]

    # for i, k in enumerate(key_map.keys()):
    #     print(key_map[holder[i + 1]])

    image_map = dict()
    for i in range(1000):
        image_map[i] = os.listdir(os.path.join(args.img_path, key_map[holder[i]]))
    
    return image_map, key_map, holder

def computeAndSaveMaps(args):
    # model = models.resnet50(pretrained=True)
    model, input_size = get_model(args.name)
    model.eval()
    model = model.cuda()

    cam = get_cam(args.cam, model, args.target_layer, args)
    image_map, key_map, label_map = get_image_list(args)
    for idx in tqdm(range(1000)):
                
        cls = key_map[label_map[idx]]
        savedir = '{}_{}'.format(key_map[label_map[idx]], label_map[idx])
        
        for file in image_map[idx]:
            result = 'T'
            result_org = 'T'

            os.makedirs(os.path.join(args.save_path, savedir), exist_ok=True)

            image, rawImage = loadImage(os.path.join(args.img_path, cls, file), imageSize=224)
            image = torch.unsqueeze(image, dim=0).cuda()
            
            if args.ours:
                # saliencyMap = ucag(cam, image, idx, args=args)
                saliencyMap = multi_scale_fusion(cam, image, idx, args=args)
            else:
                if args.img_size != 224:
                    image_ = torch.nn.functional.interpolate(image, size=(args.img_size, args.img_size), mode='bicubic', align_corners=False)
                else:
                    image_ = image
                saliencyMap, pred = cam(image_, class_idx=idx, return_logit=True, wonorm=False, image_size=(224, 224))

            # with torch.no_grad():
            pred = model(image * saliencyMap.detach())
            preds = torch.nn.functional.softmax(pred, dim=1).argmax(dim=1).item()
            if preds != idx:
                result = 'F'
            model.zero_grad()
            name, ext = file.split('.')
            if args.cam in ['rap' ,'rsp', 'agf']:
                saliencyMap = saliencyMap[0].permute(1, 2, 0).data.cpu().numpy()
                render.visualize(saliencyMap.reshape([1, saliencyMap.shape[0], saliencyMap.shape[1], 1]),
                                 os.path.join(args.save_path, savedir, '{}_{}_{:.4f}.{}'.format(name, result, pred[0, idx].item(), ext)))
            else:
                saliencyMap = saliencyMap.detach().cpu().squeeze(0).squeeze(0)
                saveMapWithColorMap(os.path.join(args.save_path, savedir, '{}_{}_{:.4f}.{}'.format(name, result, pred[0, idx].item(), ext)), saliencyMap, rawImage)
            # torch.save(saliencyMap, os.path.join(args.save_path, savedir, result + '_' + ''.join([name for name in file.split('.') + ['.pt']])))

def get_label_list(args):
    label_map_file = open('data/imagenet_labelmap.txt', 'r')
    lines = label_map_file.readlines()
    key_map = dict()
    for line in lines:
        cls_str, _, cls_name = line.replace('\n', '').split(' ')
        key_map[cls_str] = cls_name
    original_labelmap = get_labelmap()
    holder = dict()
    for k, v in original_labelmap.items():
        holder[v.split(',')[0]] = k


    image_map = dict()
    return image_map, key_map, holder


def computeAndSaveMaps_single(args):
    # model = models.resnet50(pretrained=True)
    model, input_size = get_model(args.name)
    model.eval()
    model = model.cuda()
    print(args.cam)
    cam = get_cam(args.cam, model, args.target_layer, args)
    # file = "./cat_dog.png"
    image_map, key_map, label_map = get_label_list(args)
    *_, class_str = os.path.split(args.img_path)
    class_str = class_str.split('/')[-1]
    idx = label_map[key_map[class_str]]
    
    args.img_path = 'data/val'
        
    image_map, key_map, label_map = get_image_list(args)
    cls = key_map[label_map[idx]]
    savedir = '{}_{}'.format(key_map[label_map[idx]], label_map[idx])
    
    for file in image_map[idx]:
        result = 'T'
        result_org = 'T'

        os.makedirs(os.path.join(args.save_path, savedir), exist_ok=True)

        image, rawImage = loadImage(os.path.join(args.img_path, cls, file), imageSize=224)
        image = torch.unsqueeze(image, dim=0).cuda()
        
        if args.ours:
            # saliencyMap = ucag(cam, image, idx, args=args)
            saliencyMap = multi_scale_fusion(cam, image, idx, args=args)
        else:
            if args.img_size != 224:
                image_ = torch.nn.functional.interpolate(image, size=(args.img_size, args.img_size), mode='bicubic', align_corners=False)
            else:
                image_ = image
            saliencyMap, pred = cam(image_, class_idx=idx, return_logit=True, wonorm=False, image_size=(224, 224))

        # with torch.no_grad():
        pred = model(image * saliencyMap.detach())
        preds = torch.nn.functional.softmax(pred, dim=1).argmax(dim=1).item()
        if preds != idx:
            result = 'F'
        model.zero_grad()
        name, ext = file.split('.')
        if args.cam in ['rap' ,'rsp', 'agf']:
            saliencyMap = saliencyMap[0].permute(1, 2, 0).data.cpu().numpy()
            render.visualize(saliencyMap.reshape([1, saliencyMap.shape[0], saliencyMap.shape[1], 1]),
                                os.path.join(args.save_path, savedir, '{}_{}_{:.4f}.{}'.format(name, result, pred[0, idx].item(), ext)))
        else:
            saliencyMap = saliencyMap.detach().cpu().squeeze(0).squeeze(0)
            saveMapWithColorMap(os.path.join(args.save_path, savedir, '{}_{}_{:.4f}.{}'.format(name, result, pred[0, idx].item(), ext)), saliencyMap, rawImage)


if __name__ == '__main__':
    args = parsing()
    if args.dir:
        computeAndSaveMaps_single(args)
    else:
        computeAndSaveMaps(args)