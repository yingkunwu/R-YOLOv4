import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kfloss import KFLoss
from .gaussian_dist_loss import GDLoss
from detectron2.layers.rotated_boxes import pairwise_iou_rotated

from lib.general import norm_angle


class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
    

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


def bbox_xywha_ciou(pred_boxes, target_boxes):
    # Reference: https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/86a370aa2cadea6ba7e5dffb2efc4bacc4c863ea/
    #            utils/box/box_utils.py#L47
    """
    :param pred_boxes: [num_of_objects, 4], boxes predicted by yolo and have been scaled
    :param target_boxes: [num_of_objects, 4], ground truth boxes and have been scaled
    :return: ciou loss
    """
    assert pred_boxes.size() == target_boxes.size(), "pred: {}, target: {}".format(pred_boxes.shape, target_boxes.shape)

    # xywha -> xyxya
    # xy is center point, so to get the former x of the bbox, you need to minus the 0.5 * width or height
    pred_boxes = torch.cat(
        [pred_boxes[..., :2] - pred_boxes[..., 2:4] / 2, 
         pred_boxes[..., :2] + pred_boxes[..., 2:4] / 2,
         pred_boxes[..., 4:]], dim=-1)
    target_boxes = torch.cat(
        [target_boxes[..., :2] - target_boxes[..., 2:4] / 2,
         target_boxes[..., :2] + target_boxes[..., 2:4] / 2,
         target_boxes[..., 4:]], dim=-1)

    w1 = pred_boxes[:, 2] - pred_boxes[:, 0] # x2 - x1
    h1 = pred_boxes[:, 3] - pred_boxes[:, 1] # y2 - y1
    w2 = target_boxes[:, 2] - target_boxes[:, 0]
    h2 = target_boxes[:, 3] - target_boxes[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (pred_boxes[:, 2] + pred_boxes[:, 0]) / 2
    center_y1 = (pred_boxes[:, 3] + pred_boxes[:, 1]) / 2
    center_x2 = (target_boxes[:, 2] + target_boxes[:, 0]) / 2
    center_y2 = (target_boxes[:, 3] + target_boxes[:, 1]) / 2

    inter_max_xy = torch.min(pred_boxes[:, 2:4], target_boxes[:, 2:4])
    inter_min_xy = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
    out_max_xy = torch.max(pred_boxes[:, 2:4], target_boxes[:, 2:4])
    out_min_xy = torch.min(pred_boxes[:, :2], target_boxes[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2) # c ^ 2
    union = area1 + area2 - inter_area
    u = inter_diag / (outer_diag + 1e-15)

    iou = inter_area / (union + 1e-15)
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)

    # alpha is a constant, it don't have gradient
    with torch.no_grad():
        S = 1 - iou
        alpha = v / (S + v)

    ciou = iou - (u + alpha * v)
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)

    ariou = torch.abs(torch.cos(pred_boxes[:, 4] - target_boxes[:, 4])) * iou

    return ariou, ciou


class ComputeLoss:
    def __init__(self, model, hyp):
        device = next(model.parameters()).device  # get model device

        self.ignore_thresh = hyp['ignore_thresh']
        self.lambda_coord = hyp['box']
        self.lambda_conf_scale = hyp['obj']
        self.lambda_cls_scale = hyp['cls']
        self.kfloss = KFLoss()
        self.gdloss = GDLoss(reduction = "none")

        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['obj_pw']], device=device))
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['cls_pw']], device=device))

        g = hyp['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls = FocalLoss(BCEcls, g)
            BCEobj = FocalLoss(BCEobj, g)

        self.BCEobj = BCEobj
        self.BCEcls = BCEcls
        self.delta = 0.4

        self.rotated_anchors = model.rotated_anchors
        self.anchors = model.anchors
        self.angles = model.angles
        self.strides = model.strides

    def __call__(self, outputs, target):
        device = target.device

        # initializing loss
        reg_loss, conf_loss, cls_loss, xy_loss, kf_loss = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        angles = torch.tensor(self.angles, device=device)
        
        for i, out in enumerate(outputs):
            # num of (batches, anchors(3*6), downsample grid sizes, _ , classes)
            nB, nA, nG, _, nC = out.size()
            rotated_anchors = torch.tensor(self.rotated_anchors[i], device=device)
            anchors = torch.tensor(self.anchors[i], device=device) / self.strides[i]

            obj_mask, noobj_mask, tbbox, tcls = self.build_targets(target, anchors, angles, rotated_anchors, nB, nA, nG, nC - 6, device)

            tconf = torch.zeros_like(out[..., 0], device=device)  # target obj

            # --------------------
            # - Calculating Loss -
            # --------------------

            if len(target) > 0:
                anchor_wh = rotated_anchors[:, :2].view([1, -1, 1, 1, 2])

                pxy = out[..., 0:2].sigmoid() * 2 - 0.5
                pwh = torch.exp(out[..., 2:4]) * anchor_wh
                pa = (out[..., 4].sigmoid() - 0.5) * 0.5236 # predicted angle

                pbbox = torch.cat((pxy, pwh, pa.unsqueeze(-1)), -1)  # predicted box

                pconf = out[..., 5] # objectness score
                pcls = out[..., 6:] # confidence score of classses

                # reg_loss += (reg_magnitude * reg_vector).mean()
                reg, xy, kf, KFIoU = self.kfloss(pbbox[obj_mask], tbbox[obj_mask])

                GDLoss = self.gdloss(pbbox[obj_mask], tbbox[obj_mask])
                #TODO Riou
                riou_o, _ = pairwise_iou_rotated(pbbox[obj_mask], tbbox[obj_mask]).max(1)

                Vp = self.delta * GDLoss
                alpha  = Vp / ( Vp - riou_o + 1)
                riou = (riou_o - alpha * Vp).clamp(0)  # rotated intersection over union
                tconf[obj_mask] = riou

                reg_loss += reg
                xy_loss += xy 
                kf_loss += kf 

                # BCE Loss for object's prediction
                conf_loss += self.BCEobj(pconf[obj_mask], tconf[obj_mask])

                # Binary Cross Entropy Loss for class' prediction
                cls_loss += self.BCEcls(pcls[obj_mask], tcls[obj_mask])

            conf_loss += self.BCEobj(pconf[noobj_mask], tconf[noobj_mask])

        # Loss scaling
        reg_loss = self.lambda_coord * reg_loss
        conf_loss = self.lambda_conf_scale * conf_loss
        cls_loss = self.lambda_cls_scale * cls_loss
        loss = reg_loss + conf_loss + cls_loss

        # --------------------
        # -   Logging Info   -
        # --------------------
        loss_items = {
            "total_loss": loss.detach().cpu().item(),
            "kf_loss": kf_loss.detach().cpu().item(),
            "xy_loss" : xy_loss.detach().cpu().item(),
            "conf_loss": conf_loss.detach().cpu().item(),
            "cls_loss": cls_loss.detach().cpu().item()
        }

        return loss, loss_items

    def build_targets(self, target, anchors, angles, masked_anchors, nB, nA, nG, nC, device):
        # Output tensors
        obj_mask = torch.zeros((nB, nA, nG, nG), device=device)
        noobj_mask = torch.ones((nB, nA, nG, nG), device=device)
        tbbox = torch.zeros((nB, nA, nG, nG, 5), device=device)
        tcls = torch.zeros((nB, nA, nG, nG, nC), device=device)

        # target_boxes (x, y, w, h), originally normalize w.r.t grids
        # Convert ground truth position to position that relative to the size of box (grid size)
        gxy = target[:, 2:4] * nG
        gwh = target[:, 4:6] * nG
        ga = target[:, 6]

        # Get anchors with best iou and their angle difference with ground truths
        ious = []
        diffs = []

        with torch.no_grad():
            for i in range(0, len(anchors), 2):
                anchor = anchors[i:i + 2]
                iou = anchor_wh_iou(anchor, gwh)
                ious.append(iou)
            for angle in angles:
                diff = torch.abs(torch.cos(angle - ga))
                diffs.append(diff)
            ious = torch.stack(ious)
            diffs = torch.stack(diffs)
        _, best_i = ious.max(0) #choose the anchor first
        _, best_d = diffs.max(0) #then choose the angle
        best_n = best_i * 6 + best_d

        # Separate target values
        # b indicates which batch, target_labels is the class label (0 or 1)
        b, target_labels = target[:, :2].long().t()
        gij = gxy.long()
        gi, gj = gij.t()

        # Avoid the error caused by the wrong position of the center coordinate of objects
        gi = torch.clamp(gi, 0, nG - 1)
        gj = torch.clamp(gj, 0, nG - 1)
        ga = norm_angle(ga - masked_anchors[best_n][:, 2])

        # Bounding Boxes
        tbbox[b, best_n, gj, gi] = torch.cat((gxy - gij, gwh, ga.unsqueeze(-1)), -1)

        # Set masks to specify object's location, for img the row is y and col is x
        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0

        # One-hot encoding of label
        tcls[b, best_n, gj, gi, target_labels] = 1

        obj_mask = obj_mask.type(torch.bool)
        noobj_mask = noobj_mask.type(torch.bool)

        return obj_mask, noobj_mask, tbbox, tcls
