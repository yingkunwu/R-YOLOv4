import torch
import torch.nn as nn
import numpy as np
from model.backbone import Backbone
from model.neck import Neck
from model.head import Head
from model.yololayer import YoloCSLLayer, YoloKFIoULayer


class Yolo(nn.Module):
    def __init__(self, n_classes, model_config, mode):
        super().__init__()
        # ---------------------------------------------------------------------
        # anchors = [[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]]
        # strides = [8, 16, 32]
        # as stride = 8 -> [1.5, 2.0, 2.375, 4.5, 5.0, 3.5]
        # as stride = 16 -> [2.25, 4.6875, 4.75, 3.4375, 4.5, 9.125]
        # as stride = 32 -> [4.4375, 3.4375, 6.0, 7.59375, 14.34375, 12.53125]
        # ---------------------------------------------------------------------
        anchors = model_config["anchors"]
        angles = [a * np.pi / 180 for a in model_config["angles"]]
        strides = [8, 16, 32]

        if mode == "csl":
            output_ch = (4 + 180 + 1 + n_classes) * 3
            an = self._make_anchors(strides, anchors)
            YoloLayer = YoloCSLLayer(n_classes, an, strides)
        elif mode == "kfiou":
            output_ch = (5 + 1 + n_classes) * 3 * 6
            an = self._make_rotated_anchors(strides, anchors, angles)
            YoloLayer = YoloKFIoULayer(n_classes, an, strides)
        else:
            raise NotImplementedError("Loss mode : {} not found.".format(mode))
        
        self.anchors = an
        self.nc = n_classes

        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head(output_ch)
        self.yolo = YoloLayer

    def forward(self, i, training):
        d3, d4, d5 = self.backbone(i)
        x20, x13, x6 = self.neck(d5, d4, d3)
        x2, x10, x18 = self.head(x20, x13, x6)
        out = self.yolo([x2, x10, x18], training)

        return out
    
    @staticmethod
    def _make_anchors(strides, anchors):
        an = []
        for stride, anchor in zip(strides, anchors):
            tmp = []
            for i in range(0, len(anchor), 2):
                tmp.append([anchor[i] / stride, anchor[i + 1] / stride])
            an.append(tmp)
        return an
    
    @staticmethod
    def _make_rotated_anchors(strides, anchors, angles):
        an = []
        for stride, anchor in zip(strides, anchors):
            tmp = []
            for i in range(0, len(anchor), 2):
                for angle in angles:
                    tmp.append([anchor[i] / stride, anchor[i + 1] / stride, angle])
            an.append(tmp)
        return an
