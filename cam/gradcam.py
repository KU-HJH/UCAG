import torch
import torchvision
import torch.nn.functional as F
from cam import BaseCAM

def get_backward_gradient(pred_y, y, normalize=True):
    """Returns a gradient tensor for the pointing game.

    Args:
        pred_y (:class:`torch.Tensor`): 4D tensor that the model outputs.
        y (int): target label.
        normalize (bool): If True, normalize the gradient tensor s.t. it
            sums to 1. Default: ``True``.

    Returns:
        :class:`torch.Tensor`: gradient tensor with the same shape as
        :attr:`pred_y`.
    """
    assert isinstance(pred_y, torch.Tensor)
    assert len(pred_y.shape) == 4 or len(pred_y.shape) == 2
    assert pred_y.shape[0] == 1
    assert isinstance(y, int)
    backward_gradient = torch.zeros_like(pred_y)
    backward_gradient[0, y] = torch.exp(pred_y[0, y])
    if normalize:
        backward_gradient[0, y] /= backward_gradient[0, y].sum()
    return backward_gradient

def resize_saliency(tensor, saliency, size, mode):
    """Resize a saliency map.

    Args:
        tensor (:class:`torch.Tensor`): reference tensor.
        saliency (:class:`torch.Tensor`): saliency map.
        size (bool or tuple of int): if a tuple (i.e., (width, height),
            resize :attr:`saliency` to :attr:`size`. If True, resize
            :attr:`saliency: to the shape of :attr:`tensor`; otherwise,
            return :attr:`saliency` unchanged.
        mode (str): mode for :func:`torch.nn.functional.interpolate`.

    Returns:
        :class:`torch.Tensor`: Resized saliency map.
    """
    if size is not False:
        if size is True:
            size = tensor.shape[2:]
        elif isinstance(size, tuple) or isinstance(size, list):
            # width, height -> height, width
            size = size[::-1]
        else:
            assert False, "resize must be True, False or a tuple."
        saliency = F.interpolate(
            saliency, size, mode=mode, align_corners=False)
    return saliency


class GradCAM(BaseCAM):
    def __init__(self, model, target_layer="module.layer4.2", without_sum=False):
        super().__init__(model, target_layer)
        self.without_sum = without_sum
        print('Running Original GradCAM')

    def forward(self, x, class_idx=None, retain_graph=False, return_logit=False, wonorm=False, image_size=None):
        b, c, h, w = x.size()

        # predication on raw x
        x.requires_grad_(True)
        logit = self.model(x)
        
        score = self._get_score(logit, class_idx)
        # grad = get_backward_gradient(logit, class_idx.item())
        
        self.model.zero_grad()
        
        logit.backward(score, retain_graph=retain_graph)
        # logit.backward(grad, retain_graph=retain_graph)
        gradients = self.gradients['value'].data
        activations = self.activations['value'].data
        b, k, u, v = activations.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        if self.without_sum:
            return weights * activations
        saliency_map = (weights * activations).sum(1, keepdim=True)
        
        if return_logit and wonorm:
            return saliency_map, logit
        elif wonorm:
            return saliency_map 
        if image_size is None:
            saliency_map = self._batch_normalize_map(saliency_map, relu=True, image_size=(h, w))
        else:
            saliency_map = self._batch_normalize_map(saliency_map, relu=True, image_size=image_size)
        if return_logit:
            return saliency_map, logit
        return saliency_map

    def __call__(self, x, class_idx=None, retain_graph=False, return_logit=False, wonorm=False, image_size=None):
        return self.forward(x, class_idx, retain_graph, return_logit=return_logit, wonorm=wonorm, image_size=image_size)



class XGradCAM(BaseCAM):
    def __init__(self, model, target_layer="module.layer4.2", pos_grad=False):
        super().__init__(model, target_layer)
        self.pos_grad = pos_grad
        print('Running Original XGradCAM')

    def forward(self, x, class_idx=None, retain_graph=False, return_logit=False, wonorm=False, image_size=None):
        b, c, h, w = x.size()

        # predication on raw x
        x.requires_grad_(True)
        logit = self.model(x)
        
        score = self._get_score(logit, class_idx)
        # grad = get_backward_gradient(logit, class_idx)
        
        self.model.zero_grad()
        logit.backward(score, retain_graph=retain_graph)
        # logit.backward(grad, retain_graph=retain_graph)
        gradients = self.gradients['value'].data
        activations = self.activations['value'].data
        b, k, u, v = activations.size()

        alpha = gradients.view(b, k, -1) * activations.view(b, k, -1)
        alpha = alpha.sum(-1)
        alpha = alpha / (activations.view(b, k, -1).sum(-1) + 1e-6)
        weights = alpha.view(b, k, 1, 1)
        saliency_map = (weights * activations).sum(1, keepdim=True)

        if return_logit and wonorm:
            return saliency_map, logit
        elif wonorm:
            return saliency_map 
        if image_size is None:
            saliency_map = self._batch_normalize_map(saliency_map, relu=True, image_size=(h, w))
        else:
            saliency_map = self._batch_normalize_map(saliency_map, relu=True, image_size=image_size)
        if return_logit:
            return saliency_map, logit
        return saliency_map

    def __call__(self, x, class_idx=None, retain_graph=False, return_logit=False, wonorm=False, image_size=None):
        return self.forward(x, class_idx, retain_graph, return_logit=return_logit, wonorm=wonorm, image_size=image_size)


class NewCAM(BaseCAM):
    def __init__(self, model, target_layer="module.layer4.2"):
        super().__init__(model, target_layer)
        self.mask = torch.nn.Parameter(torch.FloatTensor(1, 1, 7, 7))
        self.optim = torch.optim.SGD(self.mask, lr=1e-1, weight_decay=1e-4)

    def forward(self, x, class_idx=None, retain_graph=False, return_logit=False):
        b, c, h, w = x.size()

        # predication on raw x
        logit = self.model(x)

        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()
            
        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value'].data
        activations = self.activations['value'].data
        b, k, u, v = activations.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        saliency_map = (weights * activations).sum(1, keepdim=True)

        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
        if return_logit:
            return saliency_map, logit
        return saliency_map

    def __call__(self, x, class_idx=None, retain_graph=False, return_logit=False):
        return self.forward(x, class_idx, retain_graph, return_logit=return_logit)


class InitCAM(BaseCAM):
    def __init__(self, model, target_layer="module.layer4.2"):
        super().__init__(model, target_layer)
        self.random_model = torchvision.models.resnet50(pretrained=False).cuda().eval()
        self.cf_gradients = dict()
        self.cf_activations = dict()
        
        for module in self.random_model.named_modules():
            if module[0] == target_layer:
                module[1].register_forward_hook(self.cf_forward_hook)
                module[1].register_backward_hook(self.cf_backward_hook)
    
    def cf_backward_hook(self, module, grad_input, grad_output):
        self.cf_gradients['value'] = grad_output[0]
    
    def cf_forward_hook(self, module, input, output):
        self.cf_activations['value'] = output

    def forward(self, x, class_idx=None, retain_graph=False, return_logit=False):
        b, c, h, w = x.size()

        # predication on raw x
        logit = self.model(x)
        cf_logit = self.random_model(x)

        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
            cf_score = cf_logit[:, logit.max(1)[-1]].squeeze()
            
        else:
            score = logit[:, class_idx].squeeze()
            cf_score = cf_logit[:, class_idx].squeeze()
        
        self.random_model.zero_grad()
        cf_score.backward(retain_graph=retain_graph)
        cf_gradients = self.cf_gradients['value'].data
        cf_activations = self.cf_activations['value'].data

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value'].data
        activations = self.activations['value'].data
        b, k, u, v = activations.size()

        alpha = (gradients.view(b, k, -1) - cf_gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)
        saliency_map = (weights * (activations - cf_activations)).sum(1, keepdim=True)

        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
        if return_logit:
            return saliency_map, logit
        return saliency_map

    def __call__(self, x, class_idx=None, retain_graph=False, return_logit=False):
        return self.forward(x, class_idx, retain_graph, return_logit=return_logit)


class IntegratedCAM(BaseCAM):
    def __init__(self, model, target_layer="module.layer4.2"):
        super().__init__(model, target_layer)

    def forward(self, x, class_idx=None, retain_graph=False, return_logit=False, num_steps=20):
        b, c, h, w = x.size()
        
        x_baseline = torch.zeros_like(x)
        image = x.expand(num_steps, x_baseline.size(1), x_baseline.size(2), x_baseline.size(3))
        for i in range(num_steps):
            input = x_baseline + (x - x_baseline) * (i / num_steps)
    
            # predication on raw x
            x.requires_grad_(True)
            logit = self.model(input)
            score = self._get_score(logit, class_idx)
            self.model.zero_grad()
            score.backward(retain_graph=True)
            gradients = self.gradients['value'].data
            activations = self.activations['value'].data
            b, k, u, v = activations.size()
        
            alpha = gradients.view(b, k, -1).mean(2)
            weights = alpha.view(b, k, 1, 1)
            if i == 0:
                saliency_map = (weights * activations).sum(1, keepdim=True)
            else:
                saliency_map += (weights * activations).sum(1, keepdim=True)
        
        saliency_map = self._batch_normalize_map(saliency_map, relu=True, image_size=(h, w))

        if return_logit:
            return saliency_map, logit
        return saliency_map

    def __call__(self, x, class_idx=None, retain_graph=False, return_logit=False):
        return self.forward(x, class_idx, retain_graph, return_logit=return_logit)

class GradCAMpp(BaseCAM):
    def __init__(self, model, target_layer="module.layer4.2"):
        super().__init__(model, target_layer)

    def forward(self, x, class_idx=None, retain_graph=False, return_logit=False, wonorm=False, image_size=(224, 224)):
        b, c, h, w = x.size()

        # predication on raw x
        logit = self.model(x)

        score = self._get_score(logit, class_idx, nosum=True)
        
        self.model.zero_grad()
        score.sum().backward(retain_graph=retain_graph)
        gradients = self.gradients['value']  # dS/dA
        activations = self.activations['value']  # A
        b, k, u, v = activations.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                      activations.mul(gradients.pow(3)).view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num.div(alpha_denom + 1e-7)
        positive_gradients = F.relu(score.exp().view(-1, 1, 1, 1) * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        
        if wonorm:
            if return_logit:
                return saliency_map, logit
            return saliency_map
        
        saliency_map = self._batch_normalize_map(saliency_map, relu=True, image_size=image_size)
        
        if return_logit:
            return saliency_map, logit
        return saliency_map

    def __call__(self, x, class_idx=None, retain_graph=False, return_logit=False, wonorm=False, image_size=None):
        return self.forward(x, class_idx, retain_graph, return_logit=return_logit, wonorm=wonorm, image_size=image_size)


class SmoothGradCAM(BaseCAM):
    def __init__(self, model, target_layer="module.layer4.2", stdev_spread=0.15, n_samples=20, magnitude=True):
        super().__init__(model, target_layer)
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude

    def forward(self, x, class_idx=None, retain_graph=False, return_logit=False, wonorm=False, image_size=None):
        b, c, h, w = x.size()

        if class_idx is None:
            predicted_class = self.model(x).max(1)[-1]
        else:
            # predicted_class = torch.LongTensor([class_idx])
            predicted_class = class_idx if not isinstance(class_idx, int) else torch.LongTensor([class_idx])

        saliency_map = 0.0

        stdev = self.stdev_spread / (x.view(b, -1).max(-1)[0] - x.view(b, -1).min(-1)[0])
        std_tensor = torch.ones_like(x) * stdev.view(b, 1, 1, 1)

        self.model.zero_grad()
        for i in range(self.n_samples):
            x_plus_noise = torch.normal(mean=x, std=std_tensor)
            x_plus_noise.requires_grad_()
            x_plus_noise.cuda()
            logit = self.model(x_plus_noise)
            score = self._get_score(logit, class_idx)
            score.backward(retain_graph=True)

            gradients = self.gradients['value']
            if self.magnitude:
                gradients = gradients * gradients
            activations = self.activations['value']
            b, k, u, v = activations.size()

            alpha = gradients.view(b, k, -1).mean(2)
            weights = alpha.view(b, k, 1, 1)
            # weights = gradients

            saliency_map += (weights * activations).sum(1, keepdim=True)
        if wonorm:
            if return_logit:
                return saliency_map, logit
            return saliency_map
        if image_size is not None:
            saliency_map = self._batch_normalize_map(saliency_map, relu=True, image_size=image_size)
        else:
            saliency_map = self._batch_normalize_map(saliency_map, relu=True, image_size=(h, w))

        if return_logit:
            return saliency_map, logit
        return saliency_map

    def __call__(self, x, class_idx=None, retain_graph=False, return_logit=False, wonorm=False, image_size=None):
        return self.forward(x, class_idx, retain_graph, return_logit=return_logit, wonorm=wonorm, image_size=image_size)
