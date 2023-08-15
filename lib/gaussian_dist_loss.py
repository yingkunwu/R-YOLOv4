# Copyright (c) SJTU. All rights reserved.
from copy import deepcopy
import torch
from torch import nn
from .kfloss import xy_wh_r_2_xy_sigma


def xy_stddev_pearson_2_xy_sigma(xy_stddev_pearson):
    """Convert oriented bounding box from the Pearson coordinate system to 2-D
    Gaussian distribution.

    Args:
        xy_stddev_pearson (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xy_stddev_pearson.shape
    assert _shape[-1] == 5
    xy = xy_stddev_pearson[..., :2]
    stddev = xy_stddev_pearson[..., 2:4]
    pearson = xy_stddev_pearson[..., 4].clamp(min=1e-7 - 1, max=1 - 1e-7)
    covar = pearson * stddev.prod(dim=-1)
    var = stddev.square()
    sigma = torch.stack((var[..., 0], covar, covar, var[..., 1]),
                        dim=-1).reshape(_shape[:-1] + (2, 2))
    return xy, sigma


def postprocess(distance, fun='log1p', tau=1.0, reduction = "mean"):
    """Convert distance to loss.

    Args:
        distance (torch.Tensor)
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    if fun == 'log1p':
        distance = torch.log1p(distance)
    elif fun == 'sqrt':
        distance = torch.sqrt(distance.clamp(1e-7))
    elif fun == 'none':
        pass
    else:
        raise ValueError(f'Invalid non-linear function {fun}')

    if tau >= 1.0:
        out_dist = 1 - 1 / (tau + distance)
    else:
        out_dist = distance
    
    if reduction == "mean":
        return out_dist.mean()
    elif reduction == "sum":
        return out_dist.sum()
    else:
        return out_dist


class GDLoss(nn.Module):
    """Gaussian based loss.

    Args:
        loss_type (str):  Type of loss.
        representation (str, optional): Coordinate System.
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.
        alpha (float, optional): Defaults to 1.0.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    BAG_PREP = {
        'xy_stddev_pearson': xy_stddev_pearson_2_xy_sigma,
        'xy_wh_r': xy_wh_r_2_xy_sigma
    }

    def __init__(self,
                 representation='xy_wh_r',
                 fun='sqrt',
                 tau=0.0,
                 alpha=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(GDLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['log1p', 'none', 'sqrt']
        self.preprocess = self.BAG_PREP[representation]
        self.fun = fun
        self.tau = tau
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        pred = self.preprocess(pred)
        target = self.preprocess(target)

        xy_p, _ , Sigma_p = pred
        xy_t, _ , Sigma_t = target

        xy_distance = (xy_p - xy_t).square().sum(dim=-1)

        whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        whr_distance = whr_distance + Sigma_t.diagonal(
            dim1=-2, dim2=-1).sum(dim=-1)

        _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
        whr_distance = whr_distance + (-2) * (
            (_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

        distance = (xy_distance + self.alpha * self.alpha * whr_distance).clamp(1e-7).sqrt()

        if normalize:
            scale = 2 * (
                _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
            distance = distance / scale

        return postprocess(distance, fun=self.fun, tau=self.tau, reduction = self.reduction) * self.loss_weight

