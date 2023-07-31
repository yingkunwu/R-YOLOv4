import torch
import torch.nn as nn
import numpy as np
from model.backbone import Backbone
from model.neck import Neck
from model.head import Head
from model.yololayer import YoloLayer


class Yolo(nn.Module):
    def __init__(self, n_classes, model_config):
        super().__init__()
        output_ch = (5 + 1 + n_classes) * 3 * 6
        anchors = model_config["anchors"]
        angles = [a * np.pi / 180 for a in model_config["angles"]]
        strides = [8, 16, 32]
        self.scale_x_y = [1.2, 1.1, 1.05]
        self.rotated_anchors = self._make_anchors(strides, anchors, angles)
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head(output_ch)
        self.yolo1 = YoloLayer(num_classes=n_classes, anchors=self.rotated_anchors[0],
                               stride=strides[0], scale_x_y=self.scale_x_y[0])
        self.yolo2 = YoloLayer(num_classes=n_classes, anchors=self.rotated_anchors[1],
                               stride=strides[1], scale_x_y=self.scale_x_y[1])
        self.yolo3 = YoloLayer(num_classes=n_classes, anchors=self.rotated_anchors[2],
                               stride=strides[2], scale_x_y=self.scale_x_y[2])

    def forward(self, i, training):
        d3, d4, d5 = self.backbone(i)
        x20, x13, x6 = self.neck(d5, d4, d3)
        x2, x10, x18 = self.head(x20, x13, x6)

        out1 = self.yolo1(x2, training)
        out2 = self.yolo2(x10, training)
        out3 = self.yolo3(x18, training)

        x = []
        z = []
        for out in [out1, out2, out3]:
            if training:
                x.append(out)
            else:
                x.append(out[0])
                z.append(out[1])
        return x if training else (torch.cat(z, 1), x)
    
    @staticmethod
    def _make_anchors(strides, anchors, angles):
        rotated_anchors = []
        for stride, anchor in zip(strides, anchors):
            tmp = []
            for i in range(0, len(anchor), 2):
                for angle in angles:
                    tmp.append([anchor[i] / stride, anchor[i + 1] / stride, angle])
            rotated_anchors.append(tmp)
        return rotated_anchors
