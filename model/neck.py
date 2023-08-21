from model.utils import *


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

        self.conv4 = Conv(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv(1024, 512, 1, 1, 'leaky')

        # upsampling
        self.conv7 = Conv(512, 256, 1, 1, 'leaky')
        self.up1 = nn.Upsample(scale_factor=2)

        # R 85
        self.conv8 = Conv(512, 256, 1, 1, 'leaky')

        # R -1 -3
        self.conv9 = Conv(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv(512, 256, 1, 1, 'leaky')

        # upsampling
        self.conv14 = Conv(256, 128, 1, 1, 'leaky')
        self.up2 = nn.Upsample(scale_factor=2)
        # R 54
        self.conv15 = Conv(256, 128, 1, 1, 'leaky')

        # R -1 -3
        self.conv16 = Conv(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv(256, 128, 1, 1, 'leaky')

    def forward(self, x1, x2, x3):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        # SPP
        # A maximum pool is applied to a sliding kernel of size say, 1×1, 5×5, 9×9, 13×13.
        # The spatial dimension is preserved. The features maps from different kernel sizes are then
        # concatenated together as output.
        m1 = self.maxpool1(x1)
        m2 = self.maxpool2(x1)
        m3 = self.maxpool3(x1)

        x1 = torch.cat([m3, m2, m1, x1], dim=1)
        # SPP end
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)
        x1 = self.conv6(x1)

        # UP
        up1 = self.up1(self.conv7(x1))
        x2 = self.conv8(x2)
        # Fuse
        x2 = torch.cat([x2, up1], dim=1)

        x2 = self.conv9(x2)
        x2 = self.conv10(x2)
        x2 = self.conv11(x2)
        x2 = self.conv12(x2)
        x2 = self.conv13(x2)

        # UP
        up2 = self.up2(self.conv14(x2))
        x3 = self.conv15(x3)
        # Fuse
        x3 = torch.cat([x3, up2], dim=1)

        x3 = self.conv16(x3)
        x3 = self.conv17(x3)
        x3 = self.conv18(x3)
        x3 = self.conv19(x3)
        x3 = self.conv20(x3)

        return x3, x2, x1
