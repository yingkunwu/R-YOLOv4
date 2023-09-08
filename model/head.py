from model.utils import *


class Head(nn.Module):
    def __init__(self, output_ch):
        super().__init__()
        self.conv1 = Conv(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv(256, output_ch, 1, 1, 'linear', bn=False, bias=True)

        # R -4
        self.conv3 = Conv(128, 256, 3, 2, 'leaky')

        # R -1 -16
        self.conv4 = Conv(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv(512, output_ch, 1, 1, 'linear', bn=False, bias=True)

        # R -4
        self.conv11 = Conv(256, 512, 3, 2, 'leaky')

        # R -1 -37
        self.conv12 = Conv(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv14 = Conv(1024, 512, 1, 1, 'leaky')
        self.conv15 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv16 = Conv(1024, 512, 1, 1, 'leaky')
        self.conv17 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv18 = Conv(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)

    def forward(self, x3, x2, x1):
        out3 = self.conv1(x3)
        out3 = self.conv2(out3)

        x3 = self.conv3(x3)
        x3 = torch.cat([x3, x2], dim=1)

        x3 = self.conv4(x3)
        x3 = self.conv5(x3)
        x3 = self.conv6(x3)
        x3 = self.conv7(x3)
        x3 = self.conv8(x3)

        out2 = self.conv9(x3)
        out2 = self.conv10(out2)

        x3 = self.conv11(x3)
        x1 = torch.cat([x1, x3], dim=1)

        x1 = self.conv12(x1)
        x1 = self.conv13(x1)
        x1 = self.conv14(x1)
        x1 = self.conv15(x1)
        x1 = self.conv16(x1)

        x1 = self.conv17(x1)
        out1 = self.conv18(x1)

        return out3, out2, out1
    

class Headv5(nn.Module):
    def __init__(self, output_ch):
        super().__init__()
        self.conv1 = Conv(256, output_ch, 1, 1, 'linear', bn=False, bias=True)
        self.conv2 = Conv(512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        self.conv3 = Conv(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)

    def forward(self, x3, x2, x1):
        out3 = self.conv1(x3)
        out2 = self.conv2(x2)
        out1 = self.conv3(x1)

        return out3, out2, out1