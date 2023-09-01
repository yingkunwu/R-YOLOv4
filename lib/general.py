import torch
import numpy as np
import cv2 as cv
from detectron2.layers.nms import nms_rotated


def norm_angle(theta):
    """Limit the range of angles.
    Args:
        theta (torch.Tensor): shape(n,)
    Returns:
        theta (torch.Tensor): shape(n,)
    """
    theta = torch.where(theta >= np.pi / 2, theta - np.pi, theta)
    theta = torch.where(theta < -np.pi / 2, theta + np.pi, theta)

    assert torch.logical_and(-np.pi / 2 <= theta, theta < np.pi / 2).all(), \
        ("Theta of oriented bounding boxes are not within the boundary [-pi / 2, pi / 2)")
    
    return theta


def xywh2xyxy(x):
    """
    Convert (x, y, w, h) to (x1, y1, x2, y2).
    Arguments:
        x (torch.Tensor): shape(N, 4)
    Returns:
        y (torch.Tensor): shape(N, 4)
    """
    assert isinstance(x, torch.Tensor), "Input should be torch.tensors."

    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xywha2xyxyxyxy(boxes):
    """
    Convert (x, y, w, h, theta) to (x1, y1, x2, y2, x3, y3, x4, y4).
    Note that angle is in radians and positive rotations are clockwise (under image coordinate).
    Arguments:
        boxes (torch.Tensor): shape(N, 5)
    Returns:
        rboxes (torch.Tensor): shape(N, 4, 2)
    """
    num_samples = boxes.size(0)
    Rs = torch.zeros((num_samples, 2, 3))
    x, y, w, h, theta = boxes.unbind(dim=-1)

    for i in range(num_samples):
        R = cv.getRotationMatrix2D((float(x[i]), float(y[i])), float(theta[i] * 180 / np.pi), 1)
        Rs[i] = torch.from_numpy(R)

    x1, y1 = x - h / 2, y - w / 2
    x2, y2 = x + h / 2, y - w / 2
    x3, y3 = x + h / 2, y + w / 2
    x4, y4 = x - h / 2, y + w / 2

    p = torch.stack((x1, y1, x2, y2, x3, y3, x4, y4), dim=-1).reshape(-1, 4, 2)
    p = torch.cat((p, torch.ones((num_samples, 4, 1))), dim=-1)
    rboxes = torch.bmm(p, Rs.permute((0, 2, 1)))

    return rboxes


def xyxyxyxy2xywha(boxes):
    """
    Convert (x1, y1, x2, y2, x3, y3, x4, y4) to (x, y, w, h, theta).
    This function assumes that 4 vertices of bboxes [(x1, y2), (x2, y2), ... ] are in clockwise order.
    It returns a box defined in 180-degrees angular range, meaning that theta is determined by the long side (h) of the rectangle and x-axis.
    Also note that theta is in radians and the positive rotations of theta are clockwise as defined under image coordinate.
    Arguments:
        boxes (torch.Tensor): shape(N, 8)
    Returns:
        boxes (torch.Tensor): shape(N, 5)
    """
    num_samples = boxes.size(0)
    x1, y1, x2, y2, x3, y3, x4, y4 = boxes.unbind(dim=-1)

    x = (x1 + x2 + x3 + x4) / 4
    y = (y1 + y2 + y3 + y4) / 4
    w = (torch.linalg.norm(torch.stack((x2 - x3, y2 - y3), -1), dim=1) + 
        torch.linalg.norm(torch.stack((x1 - x4, y1 - y4), -1), dim=1)) / 2
    h = (torch.linalg.norm(torch.stack((x1 - x2, y1 - y2), -1), dim=1) + 
        torch.linalg.norm(torch.stack((x4 - x3, y4 - y3), -1), dim=1)) / 2
    theta = -(torch.atan2(y1 - y2, x1 - x2) + torch.atan2(y4 - y3, x4 - x3)) / 2

    # Make the height of bounding boxes always larger then it's width
    for i in range(num_samples):
        if w[i] >= h[i]:
            w[i], h[i] = h[i].clone(), w[i].clone()
            if theta[i] > 0:
                theta[i] = theta[i] - np.pi / 2
            else:
                theta[i] = theta[i] + np.pi / 2

    # ensure the range of theta span in [-np.pi / 2, np.pi / 2)
    theta = norm_angle(theta)

    return torch.stack((x, y, w, h, theta), -1)


def xywhr2xywhrsigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.
    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5)
    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution with shape (N, 2)
        wh (torch.Tensor): size of original bboxes
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution with shape (N, 2, 2)
    """

    _shape = xywhr.size()
    assert _shape[-1] == 5

    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-4, max=1e4)
    r = xywhr[..., 4]

    cos_r = torch.cos(r)
    sin_r = torch.sin(r)

    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)

    S = (0.5 * torch.diag_embed(wh)).square()

    sigma = R.bmm(S).bmm(R.permute(0, 2, 1)).reshape((_shape[0], 2, 2))

    return xy, wh, r, sigma


def post_process(predictions, conf_thres=0.5, iou_thres=0.4):
    """
    Args:
        predictions (torch.Tensor): shape(batch size, (grid_size_1^2 + grid_size_1^2, grid_size_3^2) x num_anchors, 6 + num_classes)
        e.g. : [1, ((52 x 52) + (26 x 26) + (13 x 13)) x 18, 8] with rotated anchors and 416 image size
    Returns:
        outputs (torch.Tensor): shape(batch size, 7)
        last dimension-> (x, y, w, h, theta, confidence score, class id)
    """

    # Settings
    max_wh = 4096 # min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 5000  # maximum number of boxes into torchvision.ops.nms()
    max_det = 1500

    outputs = [torch.zeros((0, 7), device=predictions.device)] * predictions.size(0)

    for batch, image_pred in enumerate(predictions):
        # Object confidence times class confidence
        image_pred[:, 6:] *= image_pred[:, 5:6]
        # Get predicted classes and the confidence score according to it
        class_confs, class_preds = image_pred[:, 6:].max(1, keepdim=True)  # class_preds-> index of classes
        # Detections matrix nx7 (xywhθ, conf, cls), θ ∈ (-pi/2, pi/2]
        dets = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Filter out confidence scores below threshold
        dets = dets[class_confs.view(-1) > conf_thres]
        # If none are remaining => process next image
        if not dets.shape[0]:
            continue
        # Sort by it
        dets = dets[dets[:, 5].argsort(descending=True)]
        if dets.shape[0] > max_nms:
            dets = dets[:max_nms]

        # non-maximum suppression
        c = dets[:, -1:] * max_wh  # classes
        rboxes = dets[:, :5].clone() 
        rboxes[:, :2] = rboxes[:, :2] + c # rboxes (offset by class)
        rboxes[:, 4] = rboxes[:, 4] / np.pi * 180 # convert radians to degrees
        scores = dets[:, 5]  # scores

        i = nms_rotated(rboxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        
        outputs[batch] = dets[i]

    return outputs
