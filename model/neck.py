from model.utils import *


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference):
        assert (x.data.dim() == 4)
        _, _, tw, th = target_size

        if inference:
            B, C, W, H = x.size()
            return x.view(B, C, W, 1, H, 1).expand(B, C, W, tw // W, H, th // H).contiguous().view(B, C, tw, th)
        else:
            return F.interpolate(x, size=(tw, th), mode='nearest')


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv(1024, 512, 1, 1, 'leaky')
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv(512, 256, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv(512, 256, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv(256, 128, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3, inference):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        # A maximum pool is applied to a sliding kernel of size say, 1×1, 5×5, 9×9, 13×13.
        # The spatial dimension is preserved. The features maps from different kernel sizes are then
        # concatenated together as output.
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # UP
        up = self.upsample1(x7, downsample4.size(), inference)
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size(), inference)
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6
