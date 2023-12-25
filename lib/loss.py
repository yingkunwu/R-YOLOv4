import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers.rotated_boxes import pairwise_iou_rotated

from lib.general import xywhr2xywhrsigma, norm_angle


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


def bbox_ciou(pred_boxes, target_boxes):
    # Reference: https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/86a370aa2cadea6ba7e5dffb2efc4bacc4c863ea/
    #            utils/box/box_utils.py#L47
    """
    :param pred_boxes: [num_of_objects, 4], boxes predicted by yolo and have been scaled
    :param target_boxes: [num_of_objects, 4], ground truth boxes and have been scaled
    :return: ciou loss
    """
    assert pred_boxes.size() == target_boxes.size(), "pred: {}, target: {}".format(pred_boxes.shape, target_boxes.shape)

    x1, y1, w1, h1 = pred_boxes.unbind(dim=-1)
    x2, y2, w2, h2 = target_boxes.unbind(dim=-1)

    # xywh -> xyxy
    # xy is center point, so to get the former x of the bbox, you need to minus the 0.5 * width or height
    pb = torch.stack([x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2], dim=-1)
    tb = torch.stack([x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2], dim=-1)

    inter_max_xy = torch.min(pb[:, 2:], tb[:, 2:])
    inter_min_xy = torch.max(pb[:, :2], tb[:, :2])
    out_max_xy = torch.max(pb[:, 2:], tb[:, 2:])
    out_min_xy = torch.min(pb[:, :2], tb[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (x2 - x1) ** 2 + (y2 - y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2) # c ^ 2
    union = w1 * h1 + w2 * h2 - inter_area
    u = inter_diag / (outer_diag + 1e-15)

    iou = inter_area / (union + 1e-15)
    v = (4 / (np.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)

    # alpha is a constant, it don't have gradient
    with torch.no_grad():
        S = 1 - iou
        alpha = v / (S + v)

    ciou = iou - (u + alpha * v)
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)

    return ciou


class KFLoss(nn.Module):
    """Kalman filter based loss.
    ref: https://github.com/open-mmlab/mmrotate/blob/main/mmrotate/models/losses/kf_iou_loss.py

    Args:
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'
        alpha (int, optional): coefficient to control the magnitude of kfiou.
            Defaults to 3.0
    Returns:
        loss (torch.Tensor)
    """

    def __init__(self, fun='exp', alpha=3.0):
        super(KFLoss, self).__init__()
        assert fun in ['none', 'ln', 'exp']
        self.fun = fun
        self.alpha = alpha

    def forward(self, pred, target):
        """Forward function.
        Args:
            pred (torch.Tensor): Predicted convexes
            target (torch.Tensor): Corresponding gt convexes
        Returns:
            loss (torch.Tensor)
            KFIoU (torch.Tensor)
        """
        xy_p, wh_p, r_p, Sigma_p = xywhr2xywhrsigma(pred)
        xy_t, wh_t, r_t, Sigma_t = xywhr2xywhrsigma(target)

        # The first term of KLD
        diff = (xy_p - xy_t).unsqueeze(-1)
        xy_loss = torch.log(diff.permute(0, 2, 1).bmm(Sigma_t.inverse()).bmm(diff) + 1).sum(dim=-1)

        #Vb_p = wh_p[:, 0] * wh_p[:, 1]
        #Vb_t = wh_t[:, 0] * wh_t[:, 1]

        #K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
        #Sigma = Sigma_p - K.bmm(Sigma_p)

        #eig = torch.linalg.eigvals(Sigma)
        #prod = (eig[:, 0] * eig[:, 1]).real
        #prod = torch.where(prod < 0, torch.full_like(prod, 0), prod)
        #Vb = 4 * torch.sqrt(prod)

        # make sure Vb doesn't go to NAN
        #assert not torch.any(torch.isnan(Vb))

        #KFIoU = (4 - self.alpha) * Vb / (Vb_p + Vb_t - self.alpha * Vb + 1e-6)

        wp2, hp2 = wh_p[:, 0] ** 2, wh_p[:, 1] ** 2
        wt2, ht2 = wh_t[:, 0] ** 2, wh_t[:, 1] ** 2
        cos2dr, sin2dr = torch.cos(r_p - r_t) ** 2, torch.sin(r_p - r_t) ** 2

        A = torch.sqrt(1 + (wp2 * hp2) / (wt2 * ht2) + (wp2 / wt2 + hp2 / ht2) * cos2dr + (wp2 / ht2 + hp2 / wt2) * sin2dr)
        B = torch.sqrt(1 + (wt2 * ht2) / (wp2 * hp2) + (wt2 / wp2 + ht2 / hp2) * cos2dr + (wt2 / hp2 + ht2 / wp2) * sin2dr)

        KFIoU = (4 - self.alpha) / (A + B - self.alpha)

        if self.fun == 'ln':
            kf_loss = -torch.log(KFIoU + 1e-6)
        elif self.fun == 'exp':
            kf_loss = torch.exp(1 - KFIoU) - 1
        else:
            kf_loss = 1 - KFIoU

        loss = (xy_loss + kf_loss).clamp(0)

        return loss.mean(), KFIoU


class ComputeCSLLoss:
    def __init__(self, model, hyp):
        device = next(model.parameters()).device  # get model device

        self.lambda_coord = hyp['box']
        self.lambda_conf_scale = hyp['obj']
        self.lambda_cls_scale = hyp['cls']
        self.lambda_theta = 0.5
        self.gr = 1.0

        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['obj_pw']], device=device))
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['cls_pw']], device=device))
        BCEtheta = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))

        g = hyp['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEobj = FocalLoss(BCEobj, g)
            BCEcls = FocalLoss(BCEcls, g)
            BCEtheta = FocalLoss(BCEtheta, g)

        self.BCEobj = BCEobj
        self.BCEcls = BCEcls
        self.BCEtheta = BCEtheta

        self.anchors = torch.tensor(model.anchors, device=device)
        self.na = len(self.anchors[0])
        self.nl = 3
        self.nc = model.nc # number of classes

        # Logging Info
        self.loss_items = {
            "reg_loss": 0,
            'theta_loss': 0,
            "conf_loss": 0,
            "cls_loss": 0,
            "total_loss": 0
        }

    def __call__(self, outputs, target):
        device = target.device

        # initializing loss
        reg_loss, conf_loss, cls_loss = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        theta_loss = torch.zeros(1, device=device)

        tcls, tbbox, ta, tg, indices, anchors = self.build_targets(outputs, target)
        
        for i, pi in enumerate(outputs):
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tconf = torch.zeros_like(pi[..., 0], device=device)  # target obj

            # --------------------
            # - Calculating Loss -
            # --------------------

            if len(target) > 0:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                if ps.shape[0] > 0:
                    pxy = ps[..., 0:2].sigmoid() * 2 - 0.5
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                    pbbox = torch.cat((pxy, pwh), -1)  # predicted box

                    ciou = bbox_ciou(pbbox, tbbox[i])

                    reg_loss += (1.0 - ciou).mean()

                    score_iou = ciou.detach().clamp(0).type(tconf.dtype)
                    tconf[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                    if self.nc > 1:
                        pcls = ps[..., 5:5 + self.nc] # confidence score of classses
                        t = torch.zeros_like(pcls, device=device)  # targets
                        t[range(b.shape[0]), tcls[i]] = 1
                        # Binary Cross Entropy Loss for class' prediction
                        cls_loss += self.BCEcls(pcls, t)

                    # theta Classification by Circular Smooth Label
                    theta_loss += self.BCEtheta(ps[..., 5 + self.nc:], tg[i])

                    #_, ptheta = torch.max(ps[..., 5 + self.nc:], 1, keepdim=True) # θ ∈ int[0, 179]
                    #ptheta = ptheta - 90 # θ ∈ [-90, 89]

                    #prbbox = torch.cat((pbbox, ptheta), -1)
                    #trbbox = torch.cat((tbbox[i], ta[i]), -1)
                    
                    #riou = []
                    #for j in range(prbbox.shape[0]):
                    #    riou.append(pairwise_iou_rotated(prbbox[j][None, :], trbbox[j][None, :]).squeeze(0))
                    #riou = torch.cat(riou, 0)

                    #score_iou = riou.detach().clamp(0).type(tconf.dtype)
                    #tconf[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

            # Focal Loss for object's prediction
            conf_loss += self.BCEobj(pi[..., 4], tconf) # objectness score

        # Loss scaling
        reg_loss = self.lambda_coord * reg_loss
        theta_loss = self.lambda_theta * theta_loss
        conf_loss = self.lambda_conf_scale * conf_loss
        cls_loss = self.lambda_cls_scale * cls_loss
        loss = reg_loss + conf_loss + cls_loss + theta_loss

        # --------------------
        # -   Logging Info   -
        # --------------------
        self.loss_items.update({
            "reg_loss": reg_loss.detach().cpu().item(),
            'theta_loss': theta_loss.detach().cpu().item(),
            "conf_loss": conf_loss.detach().cpu().item(),
            "cls_loss": cls_loss.detach().cpu().item(),
            "total_loss": loss.detach().cpu().item()
        })

        return loss, self.loss_items
    
    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, ta, tg, indices, anch = [], [], [], [], [], []
        gain = torch.ones(188, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)

        # targets-> (na, nt, 187 + 1)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]

            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain

            if nt:
                # Matches
                # Calculate the ratio of the ground truth box dimensions and the dimensions of each anchor template.
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < 4.0  # compare

                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T

                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, -1].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            ta.append(t[:, 6:7] * 180 / np.pi) # oriented angle of target boxes
            tg.append(t[:, 7:-1]) # circular smooth label
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, ta, tg, indices, anch
    

class ComputeKFIoULoss:
    def __init__(self, model, hyp):
        device = next(model.parameters()).device  # get model device

        self.lambda_coord = hyp['box']
        self.lambda_conf_scale = hyp['obj']
        self.lambda_cls_scale = hyp['cls']

        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['obj_pw']], device=device))
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['cls_pw']], device=device))

        g = hyp['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEobj = FocalLoss(BCEobj, g)
            BCEcls = FocalLoss(BCEcls, g)

        self.BCEobj = BCEobj
        self.BCEcls = BCEcls
        self.kfloss = KFLoss()
        self.gr = 1.0

        self.anchors = torch.tensor(model.anchors, device=device)
        self.na = len(self.anchors[0])
        self.nl = 3
        self.nc = model.nc # number of classes

        # Logging Info
        self.loss_items = {
            "reg_loss": 0,
            "conf_loss": 0,
            "cls_loss": 0,
            "total_loss": 0
        }

    def __call__(self, outputs, target):
        device = target.device

        # initializing loss
        reg_loss, conf_loss, cls_loss = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        tcls, tbbox, indices, anchors = self.build_targets(outputs, target)

        for i, pi in enumerate(outputs):
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tconf = torch.zeros_like(pi[..., 0], device=device)  # target obj

            # --------------------
            # - Calculating Loss -
            # --------------------

            if len(target) > 0:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                if ps.shape[0] > 0:
                    pxy = ps[..., 0:2].sigmoid() * 2 - 0.5
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i][:, :2]
                    pa = norm_angle((ps[..., 4:5].sigmoid() - 0.5) * 1.1 + anchors[i][:, 2:]) # predicted angle
                    pbbox = torch.cat((pxy, pwh, pa), -1)  # predicted box

                    kfloss, KFIoU = self.kfloss(pbbox, tbbox[i])
                    reg_loss += kfloss

                    score_iou = KFIoU.detach().clamp(0).type(tconf.dtype)
                    tconf[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                    if self.nc > 1:
                        pcls = ps[..., 6:] # class confidence scores
                        t = torch.zeros_like(pcls, device=device)  # targets
                        t[range(b.shape[0]), tcls[i]] = 1
                        # Binary Cross Entropy Loss for class' prediction
                        cls_loss += self.BCEcls(pcls, t)

            # Focal Loss for object's prediction
            conf_loss += self.BCEobj(pi[..., 5], tconf) # objectness score

        # Loss scaling
        reg_loss = self.lambda_coord * reg_loss
        conf_loss = self.lambda_conf_scale * conf_loss
        cls_loss = self.lambda_cls_scale * cls_loss
        loss = reg_loss + conf_loss + cls_loss

        # --------------------
        # -   Logging Info   -
        # --------------------
        self.loss_items.update({
            "reg_loss": reg_loss.detach().cpu().item(),
            "conf_loss": conf_loss.detach().cpu().item(),
            "cls_loss": cls_loss.detach().cpu().item(),
            "total_loss": loss.detach().cpu().item()
        })

        return loss, self.loss_items

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(8, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)

        # targets-> (na, nt, 7 + 1)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]

            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain

            if nt:
                # Matches
                # Calculate the ratio of the ground truth box dimensions and the dimensions of each anchor template.
                r = t[:, :, 4:6] / anchors[:, None, :2]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < 4.0  # compare

                # Calculate the difference (cosine) between the angle of targets and anchor boxes
                d = torch.abs(torch.cos(t[:, :, 6:7] - anchors[:, None, 2:]))
                k = (d > 0.866).reshape(j.shape)

                t = t[torch.logical_and(j, k)]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T

                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            ga = t[:, 6:7] # oriented angle of target boxes

            # Append
            a = t[:, -1].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh, ga), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
