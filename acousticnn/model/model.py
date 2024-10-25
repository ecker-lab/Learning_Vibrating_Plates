from .resnet import ResNet, DeepONet
from .vit import VIT
from .fno import FNO
from .unet import UNet, LocalNet


def model_factory(model_name, **kwargs):
    if model_name == 'ResNet':
        return ResNet(**kwargs)
    if model_name == 'vit':
        return VIT(**kwargs)
    if model_name == "DeepONet":
        return DeepONet(**kwargs)
    if model_name == "UNet":
        return UNet(**kwargs)
    if model_name == "LocalNet":
        return LocalNet(**kwargs)
    if model_name == "FNO":
        return FNO(**kwargs)

    else:
        raise ValueError(f'Unknown model: {model_name}')
