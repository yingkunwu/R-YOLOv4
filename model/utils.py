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


class ELAN1(nn.Module):
    def __init__(self, c1, c2, e1=0.5, e2=0.5):
        # Efficient Layer Aggregation Networks
        super().__init__()
        h1 = int(c1 * e1)  # hidden channels
        h2 = int(c1 * e2)  # hidden channels

        self.cv1 = Conv(c1, h1, 1, 1, "swish")
        self.cv2 = Conv(c1, h1, 1, 1, "swish")
        self.cv3 = Conv(h1, h2, 3, 1, "swish")
        self.cv4 = Conv(h1, h2, 3, 1, "swish")
        self.cv5 = Conv(h2, h2, 3, 1, "swish")
        self.cv6 = Conv(h2, h2, 3, 1, "swish")
        self.cv7 = Conv((h1 + h2) * 2, c2, 1, 1, "swish")

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv4(self.cv3(x2))
        x4 = self.cv6(self.cv5(x3))
        return self.cv7(torch.cat((x1, x2, x3, x4), dim=1))


class ELAN2(nn.Module):
    def __init__(self, c1, c2, e1=0.5, e2=0.25):
        # Efficient Layer Aggregation Networks
        super().__init__()
        h1 = int(c1 * e1)  # hidden channels
        h2 = int(c1 * e2)  # hidden channels

        self.cv1 = Conv(c1, h1, 1, 1, "swish")
        self.cv2 = Conv(c1, h1, 1, 1, "swish")
        self.cv3 = Conv(h1, h2, 3, 1, "swish")
        self.cv4 = Conv(h2, h2, 3, 1, "swish")
        self.cv5 = Conv(h2, h2, 3, 1, "swish")
        self.cv6 = Conv(h2, h2, 3, 1, "swish")
        self.cv7 = Conv(h1 * 2 + h2 * 4, c2, 1, 1, "swish")

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        x5 = self.cv5(x4)
        x6 = self.cv6(x5)
        return self.cv7(torch.cat((x1, x2, x3, x4, x5, x6), dim=1))


class MaxConv(nn.Module):
    def __init__(self, c1, e=0.5):
        # MaxConv: Efficient Convolution Module for Backbone Network
        super().__init__()
        c_ = int(c1 * e)  # hidden channels

        self.m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cv1 = Conv(c1, c_, 1, 1, "swish")
        self.cv2 = Conv(c1, c_, 1, 1, "swish")
        self.cv3 = Conv(c_, c_, 3, 2, "swish")

    def forward(self, x):
        x1 = self.cv1(self.m(x))
        x2 = self.cv3(self.cv2(x))
        return torch.cat((x1, x2), dim=1)


class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x
    

class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x


class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=1):
        super(RepConv, self).__init__()

        self.silu = nn.SiLU()
        self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

        self.rbr_dense = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(num_features=c2),
        )

        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d( c1, c2, 1, s, 0, bias=False),
            nn.BatchNorm2d(num_features=c2),
        )

    def forward(self, inputs):
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.silu(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)


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


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, "swish")
        self.cv2 = Conv(c1, c_, 1, 1, "swish")
        self.cv3 = Conv(c_, c_, 3, 1, "swish")
        self.cv4 = Conv(c_, c_, 1, 1, "swish")
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1, "swish")
        self.cv6 = Conv(c_, c_, 3, 1, "swish")
        self.cv7 = Conv(2 * c_, c2, 1, 1, "swish")

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))
