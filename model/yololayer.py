# References: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py

import torch
import torch.nn as nn
import numpy as np


class YoloCSLLayer(nn.Module):
    def __init__(self, num_classes, anchors, stride):
        super(YoloCSLLayer, self).__init__()
        self.nc = num_classes # number of classes
        self.anchors = anchors
        self.stride = stride

    def forward(self, out, training):
        # output.shape-> [batch_size, num_anchors * (num_classes + 6), grid_size, grid_size]
        device = out[0].device

        infer_out = []
        for i in range(3):
            bs, gs = out[i].size(0), out[i].size(2) # batch_size, grid_size
            na = len(self.anchors[i]) # num_anchors

            # out.shape-> torch.Size([batch_size, num_anchors, grid_size, grid_size, num_classes + 6])
            out[i] = out[i].view(bs, na, self.nc + 185, gs, gs).permute(0, 1, 3, 4, 2).contiguous()

            # decode
            if not training:
                # 預測出來的(x, y)是相對於每個cell左上角的點，因此這邊需要由左上角往右下角配合grid_size加上對應的offset，畫出的圖才會在正確的位置上
                # grid_xy is in size of downsample grid, this is to do sth like meshgrid to access the grid coord
                grid_x = torch.arange(gs, device=device).repeat(gs, 1).view([1, 1, gs, gs, 1])
                grid_y = torch.arange(gs, device=device).repeat(gs, 1).t().view([1, 1, gs, gs, 1])
                grid_xy = torch.cat((grid_x, grid_y), -1)

                # anchor.shape-> [1, 3, 1, 1, 1]
                anchors = torch.tensor(self.anchors[i], device=device)
                anchor_wh = anchors[:, :2].view([1, na, 1, 1, 2])

                y = out[i].sigmoid()

                # Eliminate grid sensitivity: pred_xy = 2 * (pred_xy - 0.5) + 0.5 (shifting center and scaling)
                pxy = (y[..., 0:2] * 2 - 0.5 + grid_xy) * self.stride[i]
                pwh = (y[..., 2:4] * 2) ** 2 * anchor_wh * self.stride[i]
                pconf = y[..., 4:5] # objectness score
                pcls = y[..., 5:5 + self.nc] # confidence score of classses
                pa = y[..., 5 + self.nc:] # theta classes score

                _, ptheta = torch.max(pa, 4, keepdim=True) # θ ∈ int[0, 179]
                ptheta = (ptheta - 90) / 180 * np.pi # θ ∈ [-pi/2, pi/2)

                y = torch.cat((pxy, pwh, ptheta, pconf, pcls), -1)
                y = y.view(bs, -1, self.nc + 6)

                infer_out.append(y)

        return out if training else (out, torch.cat(infer_out, 1))
    

class YoloKFIoULayer(nn.Module):
    def __init__(self, num_classes, anchors, stride):
        super(YoloKFIoULayer, self).__init__()
        self.nc = num_classes # number of classes
        self.anchors = anchors
        self.stride = stride

    def forward(self, out, training):
        # output.shape-> [batch_size, num_anchors * (num_classes + 6), grid_size, grid_size]
        device = out[0].device

        infer_out = []
        for i in range(3):
            bs, gs = out[i].size(0), out[i].size(2) # batch_size, grid_size
            na = len(self.anchors[i]) # num_anchors

            # out.shape-> torch.Size([batch_size, num_anchors, grid_size, grid_size, num_classes + 6])
            out[i] = out[i].view(bs, na, self.nc + 6, gs, gs).permute(0, 1, 3, 4, 2).contiguous()

            # decode
            if not training:
                # 預測出來的(x, y)是相對於每個cell左上角的點，因此這邊需要由左上角往右下角配合grid_size加上對應的offset，畫出的圖才會在正確的位置上
                # grid_xy is in size of downsample grid, this is to do sth like meshgrid to access the grid coord
                grid_x = torch.arange(gs, device=device).repeat(gs, 1).view([1, 1, gs, gs, 1])
                grid_y = torch.arange(gs, device=device).repeat(gs, 1).t().view([1, 1, gs, gs, 1])
                grid_xy = torch.cat((grid_x, grid_y), -1)

                # anchor.shape-> [1, 18, 1, 1, 1]
                anchors = torch.tensor(self.anchors[i], device=device)
                anchor_wh = anchors[:, :2].view([1, na, 1, 1, 2])
                anchor_a = anchors[:, 2].view([1, na, 1, 1, 1])

                y = out[i].sigmoid()

                # Eliminate grid sensitivity: pred_xy = 2 * (pred_xy - 0.5) + 0.5 (shifting center and scaling)
                pxy = (y[..., 0:2] * 2 - 0.5 + grid_xy) * self.stride[i]
                pwh = (y[..., 2:4] * 2) ** 2 * anchor_wh * self.stride[i]
                pa = (y[..., 4:5] - 0.5) * 0.5236 + anchor_a
                pconf = y[..., 5:6] # objectness score
                pcls = y[..., 6:] # confidence score of classses

                y = torch.cat((pxy, pwh, pa, pconf, pcls), -1)
                y = y.view(bs, -1, self.nc + 6)

                infer_out.append(y)

        return out if training else (out, torch.cat(infer_out, 1))
