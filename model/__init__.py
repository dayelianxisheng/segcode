"""
模型模块
"""
from .unet import UNet
from .pspnet import PSPNet, PSPNetWithAux
from .deeplabv3 import DeepLabV3
from .unetpp import UnetPP
from .unet3p import UNet3Plus
from .transunet import TransUNet, create_transunet, TransUNetConfig
from .danet import DANet, DANetSimple
from .res_unet_plus import ResUnetPlusPlus
from .losses import DiceLoss, FocalLoss, CombinedLoss, PSPNetLoss, DeepSupervisionLoss, UNet3PlusLoss


def create_res_unet_plus(in_channels=3, num_classes=1, filters=[32, 64, 128, 256, 512]):
    """创建ResUNet++模型"""
    return ResUnetPlusPlus(channel=in_channels, filters=filters)


__all__ = [
    'UNet', 'PSPNet', 'PSPNetWithAux', 'DeepLabV3', 'UnetPP', 'UNet3Plus', 'TransUNet',
    'create_transunet', 'TransUNetConfig', 'DANet', 'DANetSimple', 'ResUnetPlusPlus',
    'create_res_unet_plus',
    'DiceLoss', 'FocalLoss', 'CombinedLoss', 'PSPNetLoss', 'DeepSupervisionLoss', 'UNet3PlusLoss'
]
