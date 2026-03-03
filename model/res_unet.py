import torch
import torch.nn as nn

# 残差块 实现 y = F(x) + x
class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels,stride,padding):
        super(ResidualConv,self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride):
        super(Upsample,self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    def forward(self, x):
        return self.upsample(x)

class ResUnet(nn.Module):
    def __init__(self, channel=3,filters=[64,128,256,512]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
        )

        self.residual_conv1 = ResidualConv(filters[0],filters[1],2,1)
        self.residual_conv2 = ResidualConv(filters[1],filters[2],2,1)

        self.bridge = ResidualConv(filters[2],filters[3],2,1)

        self.upsample1 = Upsample(filters[3],filters[3],2,2)
        self.up_residual_conv1 = ResidualConv(filters[3]+filters[2],filters[2],1,1)

        self.upsample2 = Upsample(filters[2],filters[2],2,2)
        self.up_residual_conv2 = ResidualConv(filters[2]+filters[1],filters[1],1,1)

        self.upsample3 = Upsample(filters[1],filters[1],2,2)
        self.up_residual_conv3 = ResidualConv(filters[1]+filters[0],filters[0],1,1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0],1,kernel_size=1,stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # encode
        x1 = self.input_layer(x)+self.input_skip(x)
        x2 = self.residual_conv1(x1)
        x3 = self.residual_conv2(x2)

        # bridge
        x4 =self.bridge(x3)

        # decode
        x4 = self.upsample1(x4)
        x5 = torch.cat([x4,x3],1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample2(x6)
        x7 = torch.cat([x6,x2],1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample3(x8)
        x9 = torch.cat([x8,x1],1)

        x10 = self.up_residual_conv3(x9)
        output = self.output_layer(x10)

        return output

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = ResUnet()
    out = model(x)
    print(out.shape)



