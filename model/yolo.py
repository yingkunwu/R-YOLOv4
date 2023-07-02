import torch
import torch.nn as nn
import numpy as np
from model.backbone import Backbone
from model.neck import Neck
from model.head import Head
from model.yololayer import YoloLayer


class Yolo(nn.Module):
    def __init__(self, n_classes=80):
        super().__init__()
        output_ch = (5 + 1 + n_classes) * 3 * 6
        radian = np.pi / 180
        angles = [-radian * 60, -radian * 30, 0, radian * 30, radian * 60, radian * 90]
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head(output_ch)
        self.yolo1 = YoloLayer(num_classes=n_classes, anchors=[[12, 16], [19, 36], [40, 28]],
                               angles=angles, stride=8, scale_x_y=1.2, ignore_thresh=0.6)
        self.yolo2 = YoloLayer(num_classes=n_classes, anchors=[[36, 75], [76, 55], [72, 146]],
                               angles=angles, stride=16, scale_x_y=1.1, ignore_thresh=0.6)
        self.yolo3 = YoloLayer(num_classes=n_classes, anchors=[[142, 110], [192, 243], [459, 401]],
                               angles=angles, stride=32, scale_x_y=1.05, ignore_thresh=0.6)

    def forward(self, i, inference=False):
        d3, d4, d5 = self.backbone(i)
        x20, x13, x6 = self.neck(d5, d4, d3, inference)
        x2, x10, x18 = self.head(x20, x13, x6)

        out1, mask1 = self.yolo1(x2)
        out2, mask2 = self.yolo2(x10)
        out3, mask3 = self.yolo3(x18)

        return [out1, out2, out3], [mask1, mask2, mask3]
