# References: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py

import numpy as np
from model.loss import *


def to_cpu(tensor):
    return tensor.detach().cpu()


def anchor_wh_iou(wh1, wh2):
    """
    :param wh1: width and height of ground truth boxes
    :param wh2: width and height of anchor boxes
    :return: iou
    """
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


class YoloLayer(nn.Module):
    def __init__(self, num_classes, anchors, angles, stride, scale_x_y, ignore_thresh):
        super(YoloLayer, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.angles = angles
        self.num_anchors = len(anchors) * len(angles)
        self.stride = stride
        self.scale_x_y = scale_x_y
        self.ignore_thresh = ignore_thresh
        # ---------------------------------------------------------------------
        # ------------------------ masked anchors size ------------------------
        # as stride = 8 -> [1.5, 2.0, 2.375, 4.5, 5.0, 3.5]
        # as stride = 16 -> [2.25, 4.6875, 4.75, 3.4375, 4.5, 9.125]
        # as stride = 32 -> [4.4375, 3.4375, 6.0, 7.59375, 14.34375, 12.53125]
        # ---------------------------------------------------------------------
        self.masked_anchors = [(a_w / self.stride, a_h / self.stride, a) for a_w, a_h in self.anchors for a in self.angles]
        self.reduction = "mean"

        self.lambda_coord = 1.0
        self.lambda_conf_scale = 10.0
        self.lambda_cls_scale = 1.0
        self.metrics = {}

    def build_targets(self, pred_boxes, pred_cls, target, masked_anchors):
        nB, nA, nG, _, nC = pred_cls.size()
        device = pred_boxes.device

        # Output tensors
        obj_mask = torch.zeros((nB, nA, nG, nG), device=device)
        noobj_mask = torch.ones((nB, nA, nG, nG), device=device)
        class_mask = torch.zeros((nB, nA, nG, nG), device=device)
        iou_scores = torch.zeros((nB, nA, nG, nG), device=device)
        skew_iou = torch.zeros((nB, nA, nG, nG), device=device)
        ciou_loss = torch.zeros((nB, nA, nG, nG), device=device)
        ta = torch.zeros((nB, nA, nG, nG), device=device)
        tcls = torch.zeros((nB, nA, nG, nG, nC), device=device)

        # Convert ground truth position to position that relative to the size of box (grid size)
        target_boxes = torch.cat((target[:, 2:6] * nG, target[:, 6:]), dim=-1)
        gxy = target_boxes[:, :2]
        gwh = target_boxes[:, 2:4]
        ga = target_boxes[:, 4]

        # Get anchors with best iou and their angle difference with ground truths
        arious = []
        offset = []
        with torch.no_grad():
            for anchor in masked_anchors:
                ariou = anchor_wh_iou(anchor[:2], gwh)
                cos = torch.abs(torch.cos(torch.sub(anchor[2], ga)))
                arious.append(ariou * cos)
                offset.append(torch.abs(torch.sub(anchor[2], ga)))
            arious = torch.stack(arious)
            offset = torch.stack(offset)

        best_ious, best_n = arious.max(0)

        # Separate target values
        b, target_labels = target[:, :2].long().t()
        gi, gj = gxy.long().t()

        # Set masks to specify object's location
        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0

        # Set noobj mask to zero where iou exceeds ignore threshold
        for i, (anchor_ious, angle_offset) in enumerate(zip(arious.t(), offset.t())):
            noobj_mask[b[i], (anchor_ious > self.ignore_thresh), gj[i], gi[i]] = 0
            # if iou is greater than 0.4 and the angle offset if smaller than 15 degrees then ignore training
            noobj_mask[b[i], (anchor_ious > 0.4) & (angle_offset < (np.pi / 12)), gj[i], gi[i]] = 0

        # Angle (encode)
        ta[b, best_n, gj, gi] = ga - masked_anchors[best_n][:, 2]

        # One-hot encoding of label
        tcls[b, best_n, gj, gi, target_labels] = 1
        tconf = obj_mask.float()

        # Calculate ciou loss
        iou, ciou = bbox_xywha_ciou(pred_boxes[b, best_n, gj, gi], target_boxes)
        with torch.no_grad():
            img_size = self.stride * nG
            bbox_loss_scale = 2.0 - 1.0 * gwh[:, 0] * gwh[:, 1] / (img_size ** 2)
        ciou = bbox_loss_scale * (1.0 - ciou)

        # magnitude for reg loss
        skew_iou[b, best_n, gj, gi] = torch.exp(1 - iou) - 1

        # unit vector for reg loss
        ciou_loss[b, best_n, gj, gi] = ciou

        # Compute label correctness and iou at best anchor
        class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
        iou_scores[b, best_n, gj, gi] = iou.detach()

        obj_mask = obj_mask.type(torch.bool)
        noobj_mask = noobj_mask.type(torch.bool)

        return iou_scores, skew_iou, ciou_loss, class_mask, obj_mask, noobj_mask, ta, tcls, tconf

    def forward(self, output, target=None):
        # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
        # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        # strides = [8, 16, 32]

        # output.shape-> [batch_size, num_anchors * (num_classes + 5), grid_size, grid_size]
        device = output.device
        batch_size, grid_size = output.size(0), output.size(2)

        # prediction.shape-> torch.Size([1, num_anchors, grid_size, grid_size, num_classes + 5])
        prediction = (
            output.view(batch_size, self.num_anchors, self.num_classes + 6, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2).contiguous()
        )

        pred_x = torch.sigmoid(prediction[..., 0]) * self.scale_x_y - (self.scale_x_y - 1) / 2
        pred_y = torch.sigmoid(prediction[..., 1]) * self.scale_x_y - (self.scale_x_y - 1) / 2
        pred_w = prediction[..., 2]
        pred_h = prediction[..., 3]
        pred_a = prediction[..., 4]
        pred_conf = torch.sigmoid(prediction[..., 5])
        pred_cls = torch.sigmoid(prediction[..., 6:])

        # grid.shape-> [1, 1, 52, 52, 1]
        # 預測出來的(pred_x, pred_y)是相對於每個cell左上角的點，因此這邊需要由左上角往右下角配合grid_size加上對應的offset，畫出的圖才會在正確的位置上
        grid_x = torch.arange(grid_size, device=device).repeat(grid_size, 1).view([1, 1, grid_size, grid_size])
        grid_y = torch.arange(grid_size, device=device).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size])

        # anchor.shape-> [1, 3, 1, 1, 1]
        masked_anchors = torch.tensor(self.masked_anchors, device=device)
        anchor_w = masked_anchors[:, 0].view([1, self.num_anchors, 1, 1])
        anchor_h = masked_anchors[:, 1].view([1, self.num_anchors, 1, 1])
        anchor_a = masked_anchors[:, 2].view([1, self.num_anchors, 1, 1])

        # decode
        pred_boxes = torch.empty((prediction[..., :5].shape), device=device)
        pred_boxes[..., 0] = (pred_x + grid_x)
        pred_boxes[..., 1] = (pred_y + grid_y)
        pred_boxes[..., 2] = (torch.exp(pred_w) * anchor_w)
        pred_boxes[..., 3] = (torch.exp(pred_h) * anchor_h)
        pred_boxes[..., 4] = pred_a + anchor_a

        output = torch.cat(
            (
                torch.cat([pred_boxes[..., :4] * self.stride, pred_boxes[..., 4:]], dim=-1).view(batch_size, -1, 5),
                pred_conf.view(batch_size, -1, 1),
                pred_cls.view(batch_size, -1, self.num_classes),
            ),
            -1,
        )

        if target is None:
            return output, 0
        else:
            iou_scores, skew_iou, ciou_loss, class_mask, obj_mask, noobj_mask, ta, tcls, tconf = self.build_targets(
                pred_boxes=pred_boxes, pred_cls=pred_cls, target=target, masked_anchors=masked_anchors
            )
            # --------------------
            # - Calculating Loss -
            # --------------------

            # Reg Loss for bounding box prediction
            iou_const = skew_iou[obj_mask]
            angle_loss = F.smooth_l1_loss(pred_a[obj_mask], ta[obj_mask], reduction="none")
            reg_loss = angle_loss + ciou_loss[obj_mask]
            with torch.no_grad():
                reg_const = iou_const / reg_loss
            reg_loss = (reg_loss * reg_const).mean()

            # Focal Loss for object's prediction
            FOCAL = FocalLoss(reduction=self.reduction)
            conf_loss = (
                    FOCAL(pred_conf[obj_mask], tconf[obj_mask])
                    + FOCAL(pred_conf[noobj_mask], tconf[noobj_mask])
            )

            # Binary Cross Entropy Loss for class' prediction
            cls_loss = F.binary_cross_entropy(pred_cls[obj_mask], tcls[obj_mask], reduction=self.reduction)

            # Loss scaling
            reg_loss = self.lambda_coord * reg_loss
            conf_loss = self.lambda_conf_scale * conf_loss
            cls_loss = self.lambda_cls_scale * cls_loss
            total_loss = reg_loss + conf_loss + cls_loss

            # --------------------
            # -   Logging Info   -
            # --------------------
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "reg_loss": to_cpu(reg_loss).item(),
                "conf_loss": to_cpu(conf_loss).item(),
                "cls_loss": to_cpu(cls_loss).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
            }

            return output, total_loss
