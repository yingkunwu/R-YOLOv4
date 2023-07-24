# References: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py

import torch
import torch.nn as nn
import numpy as np


class YoloLayer(nn.Module):
    def __init__(self, num_classes, anchors, angles, stride, scale_x_y, ignore_thresh):
        super(YoloLayer, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.angles = angles
        self.num_anchors = len(anchors) * len(angles)
        self.stride = stride
        self.scale_x_y = scale_x_y
        # ---------------------------------------------------------------------
        # ------------------------ masked anchors size ------------------------
        # as stride = 8 -> [1.5, 2.0, 2.375, 4.5, 5.0, 3.5]
        # as stride = 16 -> [2.25, 4.6875, 4.75, 3.4375, 4.5, 9.125]
        # as stride = 32 -> [4.4375, 3.4375, 6.0, 7.59375, 14.34375, 12.53125]
        # ---------------------------------------------------------------------
        # masked_anchors will have 18 anchors
        self.masked_anchors = [(a_w / self.stride, a_h / self.stride, a) for a_w, a_h in self.anchors for a in self.angles]

    def forward(self, output):
        # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
        # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        # strides = [8, 16, 32]

        # output.shape-> [batch_size, num_anchors * (num_classes + 6), grid_size, grid_size]
        device = output.device
        batch_size, grid_size = output.size(0), output.size(2)

        # prediction.shape-> torch.Size([batch_size, num_anchors, grid_size, grid_size, num_classes + 6])
        prediction = (
            output.view(batch_size, self.num_anchors, self.num_classes + 6, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2).contiguous()
        )

        # Eliminate grid sensitivity: pred_xy = scale_x_y * (pred_xy - 0.5) + 0.5 (shifting center and scaling)
        pred_x = torch.sigmoid(prediction[..., 0]) * self.scale_x_y - (self.scale_x_y - 1) / 2
        pred_y = torch.sigmoid(prediction[..., 1]) * self.scale_x_y - (self.scale_x_y - 1) / 2
        pred_w = prediction[..., 2]
        pred_h = prediction[..., 3]
        pred_a = prediction[..., 4]
        pred_conf = torch.sigmoid(prediction[..., 5]) # objectness score
        pred_cls = torch.sigmoid(prediction[..., 6:]) # confidence score of classses

        # grid.shape-> [1, 1, 52, 52, 1]
        # 預測出來的(pred_x, pred_y)是相對於每個cell左上角的點，因此這邊需要由左上角往右下角配合grid_size加上對應的offset，畫出的圖才會在正確的位置上
        # grid_xy is in size of downsample grid, this is to do sth like meshgrid to access the grid coord
        grid_x = torch.arange(grid_size, device=device).repeat(grid_size, 1).view([1, 1, grid_size, grid_size])
        grid_y = torch.arange(grid_size, device=device).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size])

        # anchor.shape-> [1, 18, 1, 1, 1]
        masked_anchors = torch.tensor(self.masked_anchors, device=device)
        anchor_w = masked_anchors[:, 0].view([1, self.num_anchors, 1, 1])
        anchor_h = masked_anchors[:, 1].view([1, self.num_anchors, 1, 1])
        anchor_a = masked_anchors[:, 2].view([1, self.num_anchors, 1, 1])

        # decode
        # pred_xy is predict within the cell, so we have to add the grid coord back
        # this is in scale of downsample img
        pred_boxes = torch.empty((prediction[..., :5].shape), device=device)
        pred_boxes[..., 0] = (pred_x + grid_x)
        pred_boxes[..., 1] = (pred_y + grid_y)
        pred_boxes[..., 2] = (torch.exp(pred_w) * anchor_w)
        pred_boxes[..., 3] = (torch.exp(pred_h) * anchor_h)
        pred_boxes[..., 4] = pred_a

        output = torch.cat(
            (
                torch.cat([pred_boxes[..., :4], pred_boxes[..., 4:]], dim=-1),
                pred_a.unsqueeze(-1),
                pred_conf.unsqueeze(-1),
                pred_cls
            ),
            -1,
        )

        return output, masked_anchors, anchor_a
