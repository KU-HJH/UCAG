
import copy

import torch
from torch.nn import functional as F

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


class CAMERAS():
    def __init__(self, model, target_layer, inputResolutions=None):
        self.model = model
        self.inputResolutions = inputResolutions

        if self.inputResolutions is None:
            self.inputResolutions = list(range(224, 1000, 100))

        self.classDict = {}
        self.probsDict = {}
        self.featureDict = {}
        self.gradientsDict = {}
        self.targetLayerName = target_layer

    def _recordActivationsAndGradients(self, inputResolution, image, classOfInterest=None):
        def forward_hook(module, input, output):
            self.featureDict[inputResolution] = (copy.deepcopy(output.clone().detach().cpu()))

        def backward_hook(module, grad_input, grad_output):
            self.gradientsDict[inputResolution] = (copy.deepcopy(grad_output[0].clone().detach().cpu()))

        for name, module in self.model.named_modules():
            if name == self.targetLayerName:
                forwardHandle = module.register_forward_hook(forward_hook)
                backwardHandle = module.register_backward_hook(backward_hook)

        logits = self.model(image)
        softMaxScore = F.softmax(logits, dim=1)
        probs, classes = softMaxScore.sort(dim=1, descending=True)
        # print('pred : ', softMaxScore.max(dim=1)[1])
        pred_idx = softMaxScore.max(dim=1)[1]

        if classOfInterest is None:
            ids = classes[:, [0]]
        else:
            ids = torch.tensor(classOfInterest).unsqueeze(dim=0).unsqueeze(dim=0).cuda()

        self.classDict[inputResolution] = ids.clone().detach().item()
        # self.probsDict[inputResolution] = probs[0, 0].clone().detach().item()

        # one_hot = torch.zeros_like(logits)
        # one_hot.scatter_(1, ids, 1.0)
        grad = get_backward_gradient(logits, classOfInterest)

        self.model.zero_grad()
        logits.backward(gradient=grad, retain_graph=False)
        forwardHandle.remove()
        backwardHandle.remove()
        del forward_hook
        del backward_hook
        return logits

    def _estimateSaliencyMap(self, classOfInterest):
        saveResolution = self.inputResolutions[0]
        groundTruthClass = self.classDict[saveResolution]
        meanScaledFeatures = None
        meanScaledGradients = None

        count = 0
        for resolution in self.inputResolutions:
            if groundTruthClass == self.classDict[resolution] or self.classDict[resolution] == classOfInterest:
                count += 1
                upSampledFeatures = F.interpolate(self.featureDict[resolution].cuda(), (saveResolution, saveResolution), mode='bilinear', align_corners=False)
                upSampledGradients = F.interpolate(self.gradientsDict[resolution].cuda(), (saveResolution, saveResolution), mode='bilinear', align_corners=False)

                if meanScaledFeatures is None:
                    meanScaledFeatures = upSampledFeatures
                else:
                    meanScaledFeatures += upSampledFeatures

                if meanScaledGradients is None:
                    meanScaledGradients = upSampledGradients
                else:
                    meanScaledGradients += upSampledGradients

        meanScaledFeatures /= count
        meanScaledGradients /= count

        fmaps = meanScaledFeatures
        grads = meanScaledGradients

        saliencyMap = torch.mul(fmaps, grads).sum(dim=1, keepdim=True)

        saliencyMap = F.relu(saliencyMap)
        B, C, H, W = saliencyMap.shape
        saliencyMap = saliencyMap.view(B, -1)
        saliencyMap -= saliencyMap.min(dim=1, keepdim=True)[0]
        saliencyMap /= saliencyMap.max(dim=1, keepdim=True)[0]
        saliencyMap = saliencyMap.view(B, C, H, W)

        return saliencyMap

    def __call__(self, image, class_idx=None, retain_graph=False, return_logit=False, wonorm=False, image_size=None):
        for index, inputResolution in enumerate(self.inputResolutions):
            if index == 0:
                upSampledImage = image.cuda()
            else:
                upSampledImage = F.interpolate(image, (inputResolution, inputResolution), mode='bicubic', align_corners=False).cuda()

            pred = self._recordActivationsAndGradients(inputResolution, upSampledImage, classOfInterest=class_idx)

        saliencyMap = self._estimateSaliencyMap(classOfInterest=class_idx)
        saliencyMap = F.interpolate(saliencyMap, size=image_size, mode='bilinear', align_corners=False)
        
        return saliencyMap

