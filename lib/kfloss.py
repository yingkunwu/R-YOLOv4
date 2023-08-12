import torch
from torch import nn


def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """

    _shape = xywhr.size()
    assert _shape[-1] == 5

    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7)
    r = xywhr[..., 4]

    cos_r = torch.cos(r)
    sin_r = torch.sin(r)

    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)

    S = (0.5 * torch.diag_embed(wh)).square()

    sigma = R.bmm(S).bmm(R.permute(0, 2, 1)).reshape((_shape[0], 2, 2))

    return xy, wh, sigma


class KFLoss(nn.Module):
    """Kalman filter based loss.
    ref: https://github.com/open-mmlab/mmrotate/blob/main/mmrotate/models/losses/kf_iou_loss.py

    Args:
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self, fun='none', reduction='mean'):
        super(KFLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['none', 'ln', 'exp']
        self.fun = fun
        self.reduction = reduction

    def forward(self, pred, target, beta=1.0, eps=1e-6):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            pred_decode (torch.Tensor): Predicted decode bboxes.
            targets_decode (torch.Tensor): Corresponding gt decode bboxes.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.

        Returns:
            loss (torch.Tensor)
        """
        xy_p, wh_p, Sigma_p = xy_wh_r_2_xy_sigma(pred)
        xy_t, wh_t, Sigma_t = xy_wh_r_2_xy_sigma(target)

        # Smooth-L1 norm
        diff = torch.abs(xy_p - xy_t)
        xy_loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta).sum(dim=-1)

        Vb_p = wh_p[:, 0] * wh_p[:, 1]
        Vb_t = wh_t[:, 0] * wh_t[:, 1]

        K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
        Sigma = Sigma_p - K.bmm(Sigma_p)
        
        #Vb = 4 * Sigma.det().sqrt()
        eig = torch.linalg.eigvals(Sigma)
        Vb = 4 * torch.sqrt((eig[:, 0] * eig[:, 1]).real)
        Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
        KFIoU = Vb / (Vb_p + Vb_t - Vb + eps)

        if self.fun == 'ln':
            kf_loss = -torch.log(KFIoU + eps)
        elif self.fun == 'exp':
            kf_loss = torch.exp(1 - KFIoU) - 1
        else:
            kf_loss = 1 - KFIoU

        loss = (xy_loss + kf_loss).clamp(0)

        return loss.mean(), xy_loss.mean(), kf_loss.mean(), KFIoU * 3
