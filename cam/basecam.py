import torch
import torch.nn.functional as F

class BaseCAM(object):
    def __init__(self, model, target_layer="module.layer4.2"):
        super(BaseCAM, self).__init__()
        self.model = model.eval()
        self.gradients = dict()
        self.activations = dict()

        for module in self.model.named_modules():
            if module[0] == target_layer:
                module[1].register_forward_hook(self.forward_hook)
                module[1].register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients['value'] = grad_output[0]

    def forward_hook(self, module, input, output):
        self.activations['value'] = output

    def forward(self, x, class_idx=None, retain_graph=False):
        raise NotImplementedError
    
    def _get_score(self, logit, class_idx, nosum=False):
        batch_enum = torch.arange(0, logit.size(0))
        if class_idx is None:
            # if logit.size(0) < 2:
            #     score = logit[0, logit.max(1)[-1]].squeeze()
            # else:
            #     score = logit[batch_enum, logit.max(1)[-1]].squeeze()
            return_logit = torch.zeros_like(logit)
            preds = logit.argmax(dim=1)
            return_logit[batch_enum, preds] = 1
            return return_logit

        elif logit.size(0) < 2:
            score = logit[0, class_idx].squeeze()
            if nosum:
                score = logit[batch_enum, class_idx]
        else:
            score = logit[batch_enum, class_idx].sum()
            # if nosum:
        return_logit = torch.zeros_like(logit)
        return_logit[batch_enum, class_idx] = 1
        # score = logit[batch_enum, class_idx]
        return return_logit
        # return score

    def _single_normalize_map(self, x, relu=True, image_size=(224, 224)):
        x.detach_()
        if relu:
            x = F.relu(x)
        x = F.interpolate(x, size=image_size, mode='bilinear', align_corners=False)
        x_min = x.min()
        x_max = x.max()
        x = (x - x_min) / (x_max - x_min)
        return x

    def _batch_normalize_map(self, x, relu=True, image_size=(224, 224)):
        x.detach_()
        if relu:
            x = F.relu(x)
        x = F.interpolate(x, size=image_size, mode='bilinear', align_corners=False)
        x_min = x.min(dim=-1)[0]
        x_min = x_min.min(dim=-1)[0].view(x.size(0), 1, 1, 1)
        x_max = x.max(dim=-1)[0]
        x_max = x_max.max(dim=-1)[0].view(x.size(0), 1, 1, 1)
        x = (x - x_min) / (x_max - x_min)

        return x

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)