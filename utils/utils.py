import torch


# CLRP
def clrp_target(output, vis_class, **kwargs):
    if vis_class == 'top':
        pred = output.data.max(1, keepdim=True)[1]
        mask = torch.zeros_like(output)
        mask.scatter_(1, pred, 1)
    elif vis_class == 'index':
        mask = torch.zeros_like(output)
        mask[:, kwargs['class_id']] = 1
    elif vis_class == 'target':
        mask = torch.zeros_like(output)
        mask.scatter_(1, kwargs['target'], 1)
    else:
        raise Exception('Invalid vis-class')

    return mask * output
