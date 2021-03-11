from model.utils import *


class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv(ch, ch, 1, 1, "mish"))
            resblock_one.append(Conv(ch, ch, 3, 1, "mish"))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSampleFirst(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleFirst, self).__init__()
        self.conv0 = Conv(in_channels, 32, 3, 1, "mish")
        self.conv1 = Conv(32, 64, 3, 2, "mish")
        self.conv2 = Conv(64, 64, 1, 1, "mish")
        self.conv4 = Conv(64, 64, 1, 1, "mish")  # 這邊從1延伸出來
        self.conv5 = Conv(64, 32, 1, 1, "mish")
        self.conv6 = Conv(32, 64, 3, 1, "mish")  # 這邊有shortcut從4連過來
        self.conv8 = Conv(64, 64, 1, 1, "mish")
        self.conv10 = Conv(128, out_channels, 1, 1, "mish")  # 這邊的input是conv2+conv8 所以有128

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv4(x1)
        x4 = self.conv5(x3)
        x5 = x3 + self.conv6(x4)
        x6 = self.conv8(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x7 = self.conv10(x6)
        return x7


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, res_blocks):
        super(DownSample, self).__init__()
        self.conv1 = Conv(in_channels, in_channels * 2, 3, 2, "mish")
        self.conv2 = Conv(in_channels * 2, in_channels, 1, 1, "mish")
        self.conv4 = Conv(in_channels * 2, in_channels, 1, 1, "mish")  # 這邊從1延伸出來
        self.resblock = ResBlock(ch=in_channels, nblocks=res_blocks)
        self.conv11 = Conv(in_channels, in_channels, 1, 1, "mish")
        self.conv13 = Conv(in_channels * 2, out_channels, 1, 1, "mish")  # 這邊的input是conv2+conv11 所以有128

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv4(x1)
        r = self.resblock(x3)
        x4 = self.conv11(r)
        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv13(x4)
        return x5


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DownSampleFirst(3, 64)
        self.down2 = DownSample(64, 128, 2)
        self.down3 = DownSample(128, 256, 8)
        self.down4 = DownSample(256, 512, 8)
        self.down5 = DownSample(512, 1024, 4)

    def forward(self, i):
        d1 = self.down1(i)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        return d3, d4, d5
