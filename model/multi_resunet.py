import torch


class Conv2d_batchnorm(torch.nn.Module):
    '''
    2D卷积层（带批归一化）

    参数:
        num_in_filters {int} -- 输入通道数
        num_out_filters {int} -- 输出通道数
        kernel_size {tuple} -- 卷积核大小
        stride {tuple} -- 卷积步长 (默认: (1, 1))
        activation {str} -- 激活函数 (默认: 'relu')
    '''

    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride=(1, 1), activation='relu'):
        super().__init__()
        self.activation = activation
        self.conv1 = torch.nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size,
                                     stride=stride, padding='same')
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        if self.activation == 'relu':
            return torch.nn.functional.relu(x)
        else:
            return x


class Multiresblock(torch.nn.Module):
    '''
    MultiRes Block（多分辨率块）

    参数:
        num_in_channels {int} -- 输入通道数
        num_filters {int} -- 对应UNet阶段的滤波器数量
        alpha {float} -- 超参数 (默认: 1.67)
    '''

    def __init__(self, num_in_channels, num_filters, alpha=1.67):
        super().__init__()
        self.alpha = alpha
        self.W = num_filters * alpha

        # 根据alpha计算各分支的滤波器数量
        filt_cnt_3x3 = int(self.W * 0.167)  # 3x3分支
        filt_cnt_5x5 = int(self.W * 0.333)  # 5x5分支
        filt_cnt_7x7 = int(self.W * 0.5)    # 7x7分支
        num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7

        # 残差连接
        self.shortcut = Conv2d_batchnorm(num_in_channels, num_out_filters, kernel_size=(1, 1), activation='None')

        # 多分辨率分支：3x3 -> 5x5 -> 7x7
        self.conv_3x3 = Conv2d_batchnorm(num_in_channels, filt_cnt_3x3, kernel_size=(3, 3), activation='relu')
        self.conv_5x5 = Conv2d_batchnorm(filt_cnt_3x3, filt_cnt_5x5, kernel_size=(3, 3), activation='relu')
        self.conv_7x7 = Conv2d_batchnorm(filt_cnt_5x5, filt_cnt_7x7, kernel_size=(3, 3), activation='relu')

        self.batch_norm1 = torch.nn.BatchNorm2d(num_out_filters)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_out_filters)

    def forward(self, x):
        # 残差分支
        shrtct = self.shortcut(x)

        # 多分辨率分支
        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)

        # 通道拼接
        x = torch.cat([a, b, c], axis=1)
        x = self.batch_norm1(x)

        # 残差连接
        x = x + shrtct
        x = self.batch_norm2(x)
        x = torch.nn.functional.relu(x)

        return x


class Respath(torch.nn.Module):
    '''
    ResPath（残差路径）

    在编码器和解码器之间使用额外的残差连接，减少语义差异

    参数:
        num_in_filters {int} -- 输入通道数
        num_out_filters {int} -- 输出通道数
        respath_length {int} -- ResPath长度
    '''

    def __init__(self, num_in_filters, num_out_filters, respath_length):

        super().__init__()

        self.respath_length = respath_length
        self.shortcuts = torch.nn.ModuleList([])
        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])

        for i in range(self.respath_length):
            if (i == 0):
                # 第一个残差块
                self.shortcuts.append(
                    Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size=(1, 1), activation='None'))
                self.convs.append(
                    Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size=(3, 3), activation='relu'))

            else:
                # 后续残差块
                self.shortcuts.append(
                    Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size=(1, 1), activation='None'))
                self.convs.append(
                    Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size=(3, 3), activation='relu'))

            self.bns.append(torch.nn.BatchNorm2d(num_out_filters))

    def forward(self, x):

        for i in range(self.respath_length):
            # 残差分支
            shortcut = self.shortcuts[i](x)

            # 主路径
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)

            # 残差连接
            x = x + shortcut
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)

        return x


class MultiResUnet(torch.nn.Module):
    '''
    MultiResUNet（多分辨率UNet）

    参数:
        input_channels {int} -- 输入图像通道数
        num_classes {int} -- 分割类别数
        alpha {float} -- 超参数 (默认: 1.67)

    返回:
        [torch model] -- MultiResUNet模型
    '''

    def __init__(self, input_channels, num_classes, alpha=1.67):
        super().__init__()

        self.alpha = alpha

        # ==================== 编码器路径 ====================
        # Encoder Stage 1
        self.multiresblock1 = Multiresblock(input_channels, 32)
        self.in_filters1 = int(32 * self.alpha * 0.167) + int(32 * self.alpha * 0.333) + int(32 * self.alpha * 0.5)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.respath1 = Respath(self.in_filters1, 32, respath_length=4)

        # Encoder Stage 2
        self.multiresblock2 = Multiresblock(self.in_filters1, 32 * 2)
        self.in_filters2 = int(32 * 2 * self.alpha * 0.167) + int(32 * 2 * self.alpha * 0.333) + int(
            32 * 2 * self.alpha * 0.5)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.respath2 = Respath(self.in_filters2, 32 * 2, respath_length=3)

        # Encoder Stage 3
        self.multiresblock3 = Multiresblock(self.in_filters2, 32 * 4)
        self.in_filters3 = int(32 * 4 * self.alpha * 0.167) + int(32 * 4 * self.alpha * 0.333) + int(
            32 * 4 * self.alpha * 0.5)
        self.pool3 = torch.nn.MaxPool2d(2)
        self.respath3 = Respath(self.in_filters3, 32 * 4, respath_length=2)

        # Encoder Stage 4
        self.multiresblock4 = Multiresblock(self.in_filters3, 32 * 8)
        self.in_filters4 = int(32 * 8 * self.alpha * 0.167) + int(32 * 8 * self.alpha * 0.333) + int(
            32 * 8 * self.alpha * 0.5)
        self.pool4 = torch.nn.MaxPool2d(2)
        self.respath4 = Respath(self.in_filters4, 32 * 8, respath_length=1)

        # Bottleneck
        self.multiresblock5 = Multiresblock(self.in_filters4, 32 * 16)
        self.in_filters5 = int(32 * 16 * self.alpha * 0.167) + int(32 * 16 * self.alpha * 0.333) + int(
            32 * 16 * self.alpha * 0.5)

        # ==================== 解码器路径 ====================
        # Decoder Stage 1
        self.upsample6 = torch.nn.ConvTranspose2d(self.in_filters5, 32 * 8, kernel_size=(2, 2), stride=(2, 2))
        self.concat_filters1 = 32 * 8 * 2
        self.multiresblock6 = Multiresblock(self.concat_filters1, 32 * 8)
        self.in_filters6 = int(32 * 8 * self.alpha * 0.167) + int(32 * 8 * self.alpha * 0.333) + int(
            32 * 8 * self.alpha * 0.5)

        # Decoder Stage 2
        self.upsample7 = torch.nn.ConvTranspose2d(self.in_filters6, 32 * 4, kernel_size=(2, 2), stride=(2, 2))
        self.concat_filters2 = 32 * 4 * 2
        self.multiresblock7 = Multiresblock(self.concat_filters2, 32 * 4)
        self.in_filters7 = int(32 * 4 * self.alpha * 0.167) + int(32 * 4 * self.alpha * 0.333) + int(
            32 * 4 * self.alpha * 0.5)

        # Decoder Stage 3
        self.upsample8 = torch.nn.ConvTranspose2d(self.in_filters7, 32 * 2, kernel_size=(2, 2), stride=(2, 2))
        self.concat_filters3 = 32 * 2 * 2
        self.multiresblock8 = Multiresblock(self.concat_filters3, 32 * 2)
        self.in_filters8 = int(32 * 2 * self.alpha * 0.167) + int(32 * 2 * self.alpha * 0.333) + int(
            32 * 2 * self.alpha * 0.5)

        # Decoder Stage 4
        self.upsample9 = torch.nn.ConvTranspose2d(self.in_filters8, 32, kernel_size=(2, 2), stride=(2, 2))
        self.concat_filters4 = 32 * 2
        self.multiresblock9 = Multiresblock(self.concat_filters4, 32)
        self.in_filters9 = int(32 * self.alpha * 0.167) + int(32 * self.alpha * 0.333) + int(32 * self.alpha * 0.5)

        # 最终输出卷积
        self.conv_final = Conv2d_batchnorm(self.in_filters9, num_classes + 1, kernel_size=(1, 1), activation='None')

    def forward(self, x):
        # ==================== 编码器 ====================
        # Encoder Stage 1
        x_multires1 = self.multiresblock1(x)
        x_pool1 = self.pool1(x_multires1)
        x_multires1 = self.respath1(x_multires1)

        # Encoder Stage 2
        x_multires2 = self.multiresblock2(x_pool1)
        x_pool2 = self.pool2(x_multires2)
        x_multires2 = self.respath2(x_multires2)

        # Encoder Stage 3
        x_multires3 = self.multiresblock3(x_pool2)
        x_pool3 = self.pool3(x_multires3)
        x_multires3 = self.respath3(x_multires3)

        # Encoder Stage 4
        x_multires4 = self.multiresblock4(x_pool3)
        x_pool4 = self.pool4(x_multires4)
        x_multires4 = self.respath4(x_multires4)

        # Bottleneck
        x_multires5 = self.multiresblock5(x_pool4)

        # ==================== 解码器 ====================
        # Decoder Stage 1: 上采样 + 跳跃连接 + 多分辨率块
        up6 = torch.cat([self.upsample6(x_multires5), x_multires4], axis=1)
        x_multires6 = self.multiresblock6(up6)

        # Decoder Stage 2
        up7 = torch.cat([self.upsample7(x_multires6), x_multires3], axis=1)
        x_multires7 = self.multiresblock7(up7)

        # Decoder Stage 3
        up8 = torch.cat([self.upsample8(x_multires7), x_multires2], axis=1)
        x_multires8 = self.multiresblock8(up8)

        # Decoder Stage 4
        up9 = torch.cat([self.upsample9(x_multires8), x_multires1], axis=1)
        x_multires9 = self.multiresblock9(up9)

        # 最终输出
        out = self.conv_final(x_multires9)

        return out


if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    model = MultiResUnet(input_channels=3, num_classes=1)
    y = model(x)
    print(y.shape)
