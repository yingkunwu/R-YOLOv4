import torch.nn.functional as F
import torch.nn as nn
import torch


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(nn.Mish())
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "swish":
            self.conv.append(nn.SiLU())
        elif activation == "linear":
            pass
        else:
            raise NotImplementedError("Acativation function not found.")

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, e=0.5, act=None):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        self.cv1 = Conv(c1, c_, 1, 1, act)
        self.cv2 = Conv(c_, c2, 3, 1, act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

class CSP(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super(CSP, self).__init__()
        c_ = int(c1 * e)  # hidden channels

        self.cv1 = Conv(c1, c_, 1, 1, "mish")
        self.cv2 = Conv(c1, c_, 1, 1, "mish")
        self.cv3 = Conv(c_, c_, 1, 1, "mish")
        self.cv4 = Conv(2 * c_, c2, 1, 1, "mish")
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, e=1.0, act="mish") for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


class C5(nn.Module):
    # 5 consecutive convolutions
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c_ = int(c1 * e)  # hidden channels

        self.cv1 = Conv(c1, c_, 1, 1, "leaky")
        self.cv2 = Conv(c_, c1, 3, 1, "leaky")
        self.cv3 = Conv(c1, c_, 1, 1, "leaky")
        self.cv4 = Conv(c_, c1, 3, 1, "leaky")
        self.cv5 = Conv(c1, c2, 1, 1, "leaky")

    def forward(self, x):
        return self.cv5(self.cv4(self.cv3(self.cv2(self.cv1(x)))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c1 * e)  # hidden channels

        self.cv1 = Conv(c1, c_, 1, 1, "swish")
        self.cv2 = Conv(c1, c_, 1, 1, "swish")
        self.cv3 = Conv(2 * c_, c2, 1, 1, "swish")
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, e=1.0, act="swish") for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP)
    # A maximum pool is applied to a sliding kernel of size say, 1×1, 5×5, 9×9, 13×13.
    # The spatial dimension is preserved. 
    # The features maps from different kernel sizes are then concatenated together as output.
    def __init__(self, c1, c2):
        super().__init__()
        c_ = c1 // 2  # hidden channels

        self.cv1 = Conv(c1, c_, 1, 1, 'leaky')
        self.cv2 = Conv(c_, c1, 3, 1, 'leaky')
        self.cv3 = Conv(c1, c_, 1, 1, 'leaky')

        self.m1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.m2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.m3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        self.cv4 = Conv(c_ * 4, c_, 1, 1, 'leaky')
        self.cv5 = Conv(c_, c1, 3, 1, 'leaky')
        self.cv6 = Conv(c1, c2, 1, 1, 'leaky')

    def forward(self, x):
        x = self.cv3(self.cv2(self.cv1(x)))
        x = torch.cat([self.m3(x), self.m2(x), self.m1(x), x], dim=1)
        x = self.cv6(self.cv5(self.cv4(x)))

        return x


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels

        self.cv1 = Conv(c1, c_, 1, 1, "swish")
        self.cv2 = Conv(c_ * 4, c2, 1, 1, "swish")
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
