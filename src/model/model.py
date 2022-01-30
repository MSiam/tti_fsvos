import torch.nn as nn

from src.model.pspnet import PSPNet

def get_model(args) -> nn.Module:
    model_type = 'pspnet'
    if hasattr(args, 'model_type'):
        model_type = args.model_type

    if model_type == 'pspnet':
        return PSPNet(args, zoom_factor=8, use_ppm=True)
    else:
        raise NotImplementedError()



