import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .backbone.ca_mit import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
except ImportError:
    from backbone.ca_mit import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super(MLP, self).__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(start_dim=2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvMoudle(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvMoudle, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.9)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SegFormerHead(nn.Module):
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(c4_in_channels, embedding_dim)
        self.linear_c3 = MLP(c3_in_channels, embedding_dim)
        self.linear_c2 = MLP(c2_in_channels, embedding_dim)
        self.linear_c1 = MLP(c1_in_channels, embedding_dim)

        self.linear_fuse = ConvMoudle(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, 1)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, x):
        c1,c2,c3,c4 = x
        n,_,h,w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n,-1,c4.shape[2],c4.shape[3])
        _c4 = F.interpolate(_c4,size=c1.size()[2:],mode='bilinear',align_corners=True)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n,-1,c3.shape[2],c3.shape[3])
        _c3 = F.interpolate(_c3,size=c1.size()[2:],mode='bilinear',align_corners=True)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n,-1,c2.shape[2],c2.shape[3])
        _c2 = F.interpolate(_c2,size=c1.size()[2:],mode='bilinear',align_corners=True)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n,-1,c1.shape[2],c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4,_c3,_c2,_c1],dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        return x

class CASegFormer(nn.Module):
    def __init__(self,num_classes = 2,phi = 'b0',pretrained = False):
        super(CASegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead(num_classes=num_classes, in_channels=self.in_channels,embedding_dim=self.embedding_dim)

    def forward(self,x):
        H, W = x.size(2), x.size(3)

        x = self.backbone.forward(x)
        x = self.decode_head.forward(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

if __name__ == '__main__':
    model = CASegFormer(num_classes=2)
    x = torch.randn(1,3,256,256)
    out = model(x)
    print(out.shape)