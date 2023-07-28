import argparse
import torch
import numpy as np
import torch.nn.functional as F
from cam import *
import torchvision.models as models
from agf.modules.utils import *
from Extrans.ViT.ViT_explanation_generator import Baselines, LRP

from Extrans.ViT.ViT_explanation_generator import Baselines, LRP
from Extrans.ViT.ViT_new import vit_base_patch16_224
from Extrans.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from Extrans.ViT.ViT_orig_LRP import vit_base_patch16_224 as vit_orig_LRP

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('cam', type=str)
    parser.add_argument('-m', '--name', type=str, default='resnet50')
    parser.add_argument('-u', '--ucag', action='store_true')
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('-c', '--crop-size', default=124, type=int)
    parser.add_argument('-w', '--workers', default=8, type=int)
    parser.add_argument('-n', '--num-units', default=6, type=int)
    parser.add_argument('-t', '--target-layer', default='layer4', type=str)
    parser.add_argument('-l', '--limit-batch-size', default=16, type=int)
    parser.add_argument('-a', '--alpha', default=2.6, type=float)
    parser.add_argument('-d', '--dilation', default=1, type=int)
    parser.add_argument('-p', '--padding', default=0, type=int)
    parser.add_argument('-s', '--save', default=None, type=str)
    parser.add_argument('--stride', default=None, type=int, nargs='+')
    parser.add_argument('--img-path', default=None, type=str)
    parser.add_argument('--save-path', default='Results/', type=str)
    parser.add_argument('--score', action='store_true')
    parser.add_argument('--divide', action='store_true')
    parser.add_argument('--negative', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--lrp-based', action='store_true')
    parser.add_argument('--is-ablation', action='store_true')
    parser.add_argument('--multi', action='store_true')
    parser.add_argument('-i', '--image-size', default=224, type=int)
    
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

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def single_normalize_map(x, relu=True, image_size=(224, 224)):
    if relu:
        x = F.relu(x)
    x_min = x.min()
    x_max = x.max()
    x = (x - x_min) / (x_max - x_min)
    return x

def batch_normalize_map(x, relu=True, image_size=None, score=None):
        x = x.detach()
        if relu:
            x = F.relu(x)
        if image_size is not None:
            x = F.interpolate(x, size=image_size, mode='bilinear', align_corners=False)
        x_min = x.min(dim=-1)[0]
        x_min = x_min.min(dim=-1)[0].view(x.size(0), 1, 1, 1)
        x_max = x.max(dim=-1)[0]
        x_max = x_max.max(dim=-1)[0].view(x.size(0), 1, 1, 1)
        x = (x - x_min) / (x_max - x_min)

        return x
    
def get_cam(name, model, target_layer='layer4', args=None):
    if name in ['layercam', 'smoothgradcam++','gradcam++',
                'sscam', 'iscam', 'scorecam']:
        import torchcam
        if name == 'xgradcam':
            print('xgradcam - torchcam')
            cam_extractor = torchcam.methods.XGradCAM(model, target_layer)
        elif name == 'layercam':
            cam_extractor = torchcam.methods.LayerCAM(model, ['layer1', 'layer2', 'layer3', 'layer4'])
        elif name == 'gradcam':
            print('gradcam - torchcam')
            cam_extractor = torchcam.methods.GradCAM(model, target_layer)
        elif name == 'gradcam++':
            print('gradcampp - torchcam')
            cam_extractor = torchcam.methods.GradCAMpp(model, target_layer)
        elif name == 'iscam':
            cam_extractor = torchcam.methods.ISCAM(model, target_layer, batch_size=2)
        elif name == 'smoothgradcam++':
            cam_extractor = torchcam.methods.SmoothGradCAMpp(model, target_layer, num_samples=4)
        elif name == 'sscam':
            cam_extractor = torchcam.methods.SSCAM(model, target_layer)
        elif name == 'scorecam':
            cam_extractor = torchcam.methods.ScoreCAM(model, target_layer, batch_size=16)
        def cam(input, class_idx, return_logit=False, wonorm=False, image_size=(224, 224)):
            # with torch.no_grad():
            if isinstance(class_idx, int):
                class_idx = class_idx
            else:
                class_idx = class_idx.tolist()
            scores = model(input)
            cams = cam_extractor(class_idx=class_idx, scores=scores.exp(), normalized = (not wonorm))
            if name == 'layercam':
                cams = [cam_extractor.fuse_cams(cams)]
            if wonorm:
                if return_logit:
                    return cams[0].unsqueeze(1), scores
                
                return cams[0].unsqueeze(1)
            cams = F.interpolate(cams[0].unsqueeze(1), size=image_size, mode='bilinear', align_corners=False)
            if return_logit:
                return cams, scores
            return cams
        return cam
    elif name == 'xgradcam':
        return XGradCAM(model, target_layer=target_layer)
    elif name == 'gradcam':
        return GradCAM(model, target_layer=target_layer)
    elif name == 'gradcam_posgrad':
        return GradCAM(model, target_layer=target_layer, pos_grad=True)
    elif name == 'gradcam++':
        return GradCAMpp(model, target_layer=target_layer)
    elif name == 'smoothgradcam':
        return SmoothGradCAM(model, target_layer=target_layer)
    elif name == 'cameras':
        return CAMERAS(model, target_layer=target_layer)
    elif name == 'igcam':
        return IntegratedCAM(model, target_layer=target_layer)
    elif name == 'scorecam':
        return ScoreCAM(model, target_layer=target_layer)
    elif name == 'eigencam':
        return EigenCAM(model, target_layer=target_layer)
    elif name == 'ig':
        return IntegratedGradients(model, target_layer=target_layer)
    elif name == 'liftcam':
        return LiftCAM(model, target_layer=target_layer)
    elif name == 'rcam':
        from RCAM_modules.resnet import resnet50 as rcam_resnet50
        rcam = rcam_resnet50(pretrained=True).cuda().eval()
        def attribute(images, class_idx, return_logit=False, wonorm=False, image_size=None):
            if isinstance(class_idx, int):
                class_idx = torch.tensor([class_idx], device=images.device)
            saliency_map, logit = rcam(images, target_layer, class_idx)
            if not wonorm:
                if image_size is None:
                    saliency_map = batch_normalize_map(saliency_map)
                else:
                    saliency_map = batch_normalize_map(saliency_map, image_size=image_size)
            if return_logit:
                return saliency_map, logit
            return saliency_map
        return attribute
    elif name == 'ablationcam':
        return AblationCAM(model, target_layer=target_layer)
    elif name in ['rap', 'agf', 'rsp', 'fullgrad', 'integrad', 'smoothgrad',
                  'lrp', 'lrp_ab', 'clrp', 'sglrp']:
        
        if name in ['rap', 'lrp', 'lrp_ab', 'clrp', 'sglrp']:
        # if name == 'rap':
            if args.name == 'resnet50':
                print('get_cam: RAP + ResNet50')
                from ATTR.resnet_rap import resnet50 as resnet50_rap
                model = resnet50_rap(pretrained=True)
            elif args.name == 'vgg19':
                print('get_cam: RAP + VGG19')
                from ATTR.vgg_rap import vgg19 as vgg19_rap
                model = vgg19_rap(pretrained=True)
                                
            model = model.eval().cuda()
            
        elif name == 'agf':
            kwargs = {
                'no_a': args.no_a,
                'no_fx': args.no_fx,
                'no_fdx': args.no_fdx,
                'no_m': args.no_m,
                'no_reg': args.no_reg,
                'gradcam': args.gradcam
            }
            if args.name == 'resnet50':
                from ATTR.resnet import resnet50 as resnet50_agf
                print('getcam: AGF + ResNet50')
                model = resnet50_agf(pretrained=True)
            elif args.name == 'vgg19':
                from ATTR.vgg import vgg19 as vgg19_agf
                print('getcam: AGF + VGG19')
                model = vgg19_agf(pretrained=True)
            model = model.eval().cuda()
        elif name == 'fullgrad':
            from ATTR.fullgrad import FullGrad
            if args.name == 'resnet50':
                print('getcam: full-grad + ResNet50')
                from ATTR.resnet_fg import resnet50 as resnet50_fg
                model = resnet50_fg(pretrained=True).cuda()
            elif args.name == 'vgg19':
                print('getcam: full-grad + VGG19')
                from ATTR.vgg_fg import vgg19 as vgg19_fg
                model = vgg19_fg(pretrained=True).cuda()
            fullgrad = FullGrad(model, torch.cuda.current_device())
        elif name == 'integrad':
            from ATTR.integrated_gradients import IntegratedGradients
            integrad = IntegratedGradients(model)
        
        elif name == 'smoothgrad':
            from ATTR.smoothgrad import VanillaBackprop, generate_smooth_grad
            model.zero_grad()
            param_n = 50
            
            param_sigma_multiplier = 4
            smoothgrads = VanillaBackprop(model)
        
        if name == 'rsp':
            if args.name == 'resnet50':
                print('get_cam: RSP + ResNet50')
                from ATTR.resnet_rsp import resnet50 as resnet50_rsp
                model = resnet50_rsp(pretrained=True).cuda().eval()
            elif args.name == 'vgg19':
                print('get_cam: RSP + VGG19')
                from ATTR.vgg_rsp import vgg19 as vgg19_rsp
                model = vgg19_rsp(pretrained=True).cuda().eval()
            
            # if args.ucag:
            # else:
            cam = GradCAM(model, target_layer=target_layer, without_sum=True)
                
            def attribute(images, class_idx, return_logit=False, wonorm=False, image_size=None):
                images.requires_grad_(True)
                predictions = model(images)
                cam_tmp = cam(images, class_idx)
                cam_tmp = cam_tmp.detach().cpu().numpy()
                cam_tmp = cam_tmp / (cam_tmp.max() + 1e-9)
                rel = torch.autograd.Variable(torch.tensor(cam_tmp)).cuda()
                RAP = model.RSP(R=rel)
                Res = (RAP).sum(dim=1, keepdim=True)
                # if wonorm:
                if return_logit:
                    return Res, predictions
                return Res
        else:
            
            def attribute(images, class_idx, return_logit=False, wonorm=False, image_size=None):
                model.zero_grad()
                images = images.requires_grad_(True)
                predictions = model(images)
                if isinstance(class_idx, int):
                    class_idx = [class_idx]
                elif isinstance(class_idx, torch.Tensor):
                    class_idx = class_idx.tolist()
                if name == 'agf':
                    # if isinstance(class_idx, torch.Tensor):
                        # class_idx = class_idx.tolist()
                    # else:
                        # class_idx = [class_idx]
                    Res = model.AGF(class_id=class_idx, **kwargs)
                    # Res = model.AGF(**kwargs)
                elif name == 'rap':
                    # T = clrp_target(predictions, 'top')
                    T = clrp_target(predictions, 'index', class_id=class_idx)
                    Res = model.RAP_relprop(R=T)
                elif name == 'lrp':
                    # T = clrp_target(predictions, 'top')
                    T = clrp_target(predictions, 'index', class_id=class_idx)
                    Res = model.relprop(R=T, alpha=1)
                    Res = Res - Res.mean()
                elif name == 'lrp_ab':
                    # T = clrp_target(predictions, 'top')
                    T = clrp_target(predictions, 'index', class_id=class_idx)
                    Res = model.relprop(R=T, alpha=2)
                elif name == 'clrp':
                    # Tt = clrp_target(predictions, 'top')
                    # To = clrp_others(predictions, 'top')
                    
                    Tt = clrp_target(predictions, 'index', class_id=class_idx[0])
                    To = clrp_others(predictions, 'index', class_id=class_idx[0])

                    clrp_rel_target = model.relprop(R=Tt, alpha=1)
                    clrp_rel_others = model.relprop(R=To, alpha=1)

                    clrp_rscale = clrp_rel_target.sum(dim=[1, 2, 3], keepdim=True) / clrp_rel_others.sum(dim=[1, 2, 3],
                                                                                                        keepdim=True)
                    Res = clrp_rel_target - clrp_rel_others * clrp_rscale
                elif name == 'sglrp':
                    # Tt = clrp_target(predictions, 'top')
                    # To = clrp_others(predictions, 'top')
                    Tt = clrp_target(predictions, 'index', class_id=class_idx[0])
                    To = clrp_others(predictions, 'index', class_id=class_idx[0])

                    sglrp_rel_target = model.relprop(R=Tt, alpha=1)
                    sglrp_rel_others = model.relprop(R=To, alpha=1)

                    sglrp_rscale = sglrp_rel_target.sum(dim=[1, 2, 3], keepdim=True) / sglrp_rel_others.sum(dim=[1, 2, 3],
                                                                                                            keepdim=True)
                    Res = sglrp_rel_target - sglrp_rel_others * sglrp_rscale
                
                elif name == 'fullgrad':
                    Res = fullgrad.saliency(images, target_class=class_idx)
                    Res = Res - Res.mean()
                elif name == 'integrad':
                    pred_class = predictions.data.max(1, keepdim=True)[1].squeeze(1).item()
                    cam = integrad.generate_integrated_gradients(images,
                                                                pred_class,
                                                                100)
                    Res = cam.abs()
                    Res = Res - Res.mean()
                elif name == 'smoothgrad':
                    pred_class = predictions.data.max(1, keepdim=True)[1].squeeze(1).item()
                    cam = generate_smooth_grad(smoothgrads,
                                            images,
                                            pred_class,
                                            param_n,
                                            param_sigma_multiplier)
                    Res = cam.abs()
                    Res = Res - Res.mean()
                    
                # print(Res.shape)
                Res = Res.sum(dim=1, keepdim=True)
                Res = F.interpolate(Res, size=(224, 224), mode='bilinear', align_corners=False)
                # Res_1 = Res.gt(0).type(Res.type())
                # Res_0 = Res.le(0).type(Res.type())
                if wonorm:
                    if return_logit:
                        return Res, predictions
                    return Res
                # Res = Res.clamp(min=0.) / Res.max()
                if return_logit:
                    return Res, predictions
                return Res
        
        return attribute
    elif name in ['rollout', 'full_lrp', 'transformer_attribution', 'lrp_last_layer', 'attn_last_layer', 'attn_gradcam']:
        model = vit_base_patch16_224(pretrained=True).cuda()
        baselines = Baselines(model)

        model_LRP = vit_LRP(pretrained=True).cuda()
        model_LRP.eval()
        lrp = LRP(model_LRP)

        # orig LRP
        model_orig_LRP = vit_orig_LRP(pretrained=True).cuda()
        model_orig_LRP.eval()
        orig_lrp = LRP(model_orig_LRP)
        model.eval()
        if name == 'rollout':
            def attribute(image, class_idx, return_logit=False, wonorm=False, image_size=None):
                if wonorm:
                    return baselines.generate_rollout(image, start_layer=1).reshape(image.size(0), 1, 14, 14)
                Res = baselines.generate_rollout(image, start_layer=1).reshape(image.size(0), 1, 14, 14)
                return torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').cuda()
        # segmentation test for the LRP baseline (this is full LRP, not partial)
        elif name == 'full_lrp':
            def attribute(image, class_idx, return_logit=False, wonorm=False, image_size=None):
                if return_logit:
                    Res, logit = orig_lrp.generate_LRP(image, method="full", index=class_idx, return_logit=return_logit)
                    Res = Res.reshape(image.size(0), 1, 224, 224)
                return Res, logit
        
        # segmentation test for our method
        elif name == 'transformer_attribution':
            def attribute(image, class_idx, return_logit=False, wonorm=False, image_size=None):
                if not isinstance(class_idx, int):
                    class_idx = class_idx.cpu().numpy()
                if wonorm:
                    if return_logit:
                        Res, logit = lrp.generate_LRP(image, index=class_idx, start_layer=1, method="transformer_attribution", return_logit=return_logit)
                        return Res.reshape(image.size(0), 1, 14, 14), logit
                    return lrp.generate_LRP(image, index=class_idx, start_layer=1, method="transformer_attribution").reshape(image.size(0), 1, 14, 14)
                Res = lrp.generate_LRP(image, index=class_idx, start_layer=1, method="transformer_attribution")
                if return_logit:
                    Res, logit = Res
                    Res = Res - Res.mean()
                    return torch.nn.functional.interpolate(Res.reshape(image.size(0), 1, 14, 14), scale_factor=16, mode='bilinear').cuda(), logit
                Res = Res.reshape(image.size(0), 1, 14, 14)
                return torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').cuda()
        
        # segmentation test for the partial LRP baseline (last attn layer)
        elif name == 'lrp_last_layer':
            def attribute(image, class_idx, return_logit=False, wonorm=False, image_size=None):
                if wonorm:
                    if return_logit:
                        Res, logit = orig_lrp.generate_LRP(image, index=class_idx, method="last_layer", is_ablation=args.is_ablation, return_logit=return_logit)
                        return Res.reshape(image.size(0), 1, 14, 14), logit

                    return orig_lrp.generate_LRP(image, index=class_idx, method="last_layer", is_ablation=args.is_ablation, return_logit=return_logit)\
                            .reshape(image.size(0), 1, 14, 14)
                Res = orig_lrp.generate_LRP(image, index=class_idx, method="last_layer", is_ablation=args.is_ablation)\
                    .reshape(image.size(0), 1, 14, 14)
                return torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').cuda()
        
        # segmentation test for the raw attention baseline (last attn layer)
        elif name == 'attn_last_layer':
            def attribute(image, class_idx, return_logit=False, wonorm=False, image_size=None):
                if wonorm:
                    if return_logit:
                        Res, logit = orig_lrp.generate_LRP(image, index=class_idx, method="last_layer_attn", is_ablation=args.is_ablation, return_logit=return_logit)
                        return Res.reshape(image.size(0), 1, 14, 14), logit

                    return orig_lrp.generate_LRP(image, index=class_idx, method="last_layer_attn", is_ablation=args.is_ablation, return_logit=return_logit)\
                    .reshape(image.size(0), 1, 14, 14)
                Res = orig_lrp.generate_LRP(image, index=class_idx, method="last_layer_attn", is_ablation=args.is_ablation, return_logit=return_logit)\
                    .reshape(image.size(0), 1, 14, 14)
                return torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').cuda()
        
        # segmentation test for the GradCam baseline (last attn layer)
        elif name == 'attn_gradcam':
            def attribute(image, class_idx, return_logit=False, wonorm=False, image_size=None):
                if wonorm:
                    if return_logit:
                        Res, logit = baselines.generate_cam_attn(image, index=class_idx, wonorm=wonorm, return_logit=return_logit)
                        return Res.reshape(image.size(0), 1, 14, 14), logit
                    return baselines.generate_cam_attn(image, index=class_idx, wonorm=wonorm, return_logit=return_logit).reshape(image.size(0), 1, 14, 14)
                Res = baselines.generate_cam_attn(image, index=class_idx, wonorm=wonorm, return_logit=return_logit).reshape(image.size(0), 1, 14, 14)
                return torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').cuda()


        def attribution(image, class_idx, return_logit=False, wonorm=False, image_size=None):
            image.requires_grad = True 
            image = image.requires_grad_()
            res = attribute(image, class_idx, return_logit, wonorm, image_size)
            # if return_logit:
            #     return res, model(image)
            return res

        return attribution


    
def get_model(name):
    input_size = 224
    if name == 'mobilenetv3' or name == 'mobilenetv3_large':
        model = models.mobilenet_v3_large(pretrained=True)
    elif name == 'mobilenetv3_small':
        model = models.mobilenet_v3_small(pretrained=True)
    elif name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif name == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif name == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif name == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif name == 'resnet152':
        model = models.resnet152(pretrained=True)
    elif name == 'wide_resnet50':
        model = models.wide_resnet50_2(pretrained=True)
    elif name == 'wide_resnet101':
        model = models.wide_resnet101_2(pretrained=True)
    elif name == 'squeezenet1_0':
        model = models.squeezenet1_0(preatrained=True)
    elif name == 'squeezenet1_1':
        model = models.squeezenet1_1(preatrained=True)
    elif name == 'inceptionv3':
        input_size = 299
        model = models.inception_v3(pretrained=True)
    elif name == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif name == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif name == 'densenet169':
        model = models.densenet169(pretrained=True)
    elif name == 'densenet201':
        model = models.densenet201(pretrained=True)
    elif name == 'shufflenetv2':
        model = models.shufflenet_v2_x1_0(pretrained=True)
    elif name == 'shufflenetv2_large':
        model = models.shufflenet_v2_x2_0(pretrained=True)
    elif name == 'vit':
        model = torch.hub.load('facebookresearch/deit:main',
                           'deit_tiny_patch16_224', pretrained=True)
    return model, input_size