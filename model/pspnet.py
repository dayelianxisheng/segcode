"""
PSPNet模型定义 - 适配二值分割
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from torchvision.models.resnet import ResNet50_Weights, ResNet101_Weights


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(ConvBnReLU, self).__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.cbr(x)


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels=512, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.pool_sizes = pool_sizes
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                ConvBnReLU(in_channels, out_channels, kernel_size=1, padding=0)
            ) for pool_size in pool_sizes
        ])
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramid_features = []
        for path in self.paths:
            pooled = path(x)
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=True)
            pyramid_features.append(upsampled)
        return torch.cat(pyramid_features, dim=1)


class PSPNet(nn.Module):
    def __init__(self, num_classes=1, backbone='resnet50', pretrained=True, use_ppm=True):
        super(PSPNet, self).__init__()
        self.num_classes = num_classes
        self.use_ppm = use_ppm
        self.backbone = backbone
        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = resnet50(weights=weights)
            in_channels = 2048
        elif backbone == 'resnet101':
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = resnet101(weights=weights)
            in_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if self.use_ppm:
            self.ppm = PyramidPoolingModule(in_channels=in_channels, out_channels=512)
            ppm_channels = 512 * 4
            final_in_channels = in_channels + ppm_channels
        else:
            final_in_channels = in_channels
        self.cls = nn.Sequential(
            ConvBnReLU(final_in_channels, 512, kernel_size=3, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        input_size = x.size()[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.use_ppm:
            ppm_features = self.ppm(x)
            x = torch.cat([x, ppm_features], dim=1)
        x = self.cls(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return self.sigmoid(x)


class PSPNetWithAux(nn.Module):
    def __init__(self, num_classes=1, backbone='resnet50', pretrained=True):
        super(PSPNetWithAux, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = resnet50(weights=weights)
            in_channels = 2048
            aux_channels = 1024
        elif backbone == 'resnet101':
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = resnet101(weights=weights)
            in_channels = 2048
            aux_channels = 1024
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.aux_cls = nn.Sequential(
            ConvBnReLU(aux_channels, 256, kernel_size=3, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        self.ppm = PyramidPoolingModule(in_channels=in_channels, out_channels=512)
        ppm_channels = 512 * 4
        final_in_channels = in_channels + ppm_channels
        self.cls = nn.Sequential(
            ConvBnReLU(final_in_channels, 512, kernel_size=3, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        input_size = x.size()[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        aux_out = self.aux_cls(x)
        aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=True)
        aux_out = self.sigmoid(aux_out)
        x = self.layer4(x)
        ppm_features = self.ppm(x)
        x = torch.cat([x, ppm_features], dim=1)
        x = self.cls(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        x = self.sigmoid(x)
        return x, aux_out
