import torch
import numpy as np
import cv2 as cv
from detectron2.layers.nms import nms_rotated


def xywh2xyxy(x):
    """
    Convert (x, y, w, h) to (x1, y1, x2, y2).
    Arguments:
        x (Tensor[N, 4])
    Returns:
        y (Tensor[N, 4])
    """
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xywha2xyxyxyxy(box):
    """
    Convert (x, y, w, h, a) to (x1, y1, x2, y2, x3, y3, x4, y4).
    Note that angle is in radians and positive rotations are clockwise (under image coordinate).
    Arguments:
        box (numpy array[5])
    Returns:
        box (numpy array[8])
    """
    x, y, degree = float(box[0]), float(box[1]), float(box[4] / np.pi * 180)
    M = cv.getRotationMatrix2D((x, y), degree, 1)

    w, h = box[2], box[3]
    x1, y1 = x - h / 2, y - w / 2
    x2, y2 = x + h / 2, y - w / 2
    x3, y3 = x + h / 2, y + w / 2
    x4, y4 = x - h / 2, y + w / 2
    p = np.array([[x1, y1, 1.0], [x2, y2, 1.0], [x3, y3, 1.0], [x4, y4, 1.0]])
    
    rbox = M @ p.T

    return rbox.T


def xyxyxyxy2xywha(box):
    """
    Convert (x1, y1, x2, y2, x3, y3, x4, y4) to (x, y, w, h, a).
    This function assumes that 4 vertices of bboxes [(x1, y2), (x2, y2), ... ] are in clockwise order.
    It returns a box defined in 180-degrees angular range, meaning that theta is determined by the long side (h) of the rectangle and x-axis.
    Also note that theta is in radians and the positive rotations of theta are clockwise as defined under image coordinate.
    Arguments:
        box (Tensor[N, 8])
    Returns:
        z (Tensor[N, 5])
    """
    num_samples = box.size(0)
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:, 0], box[:, 1], box[:, 2], box[:, 3], box[:, 4], box[:, 5], box[:, 6], box[:, 7]

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
            tmp1, tmp2 = h[i].clone(), w[i].clone()
            w[i], h[i] = tmp1, tmp2
            if theta[i] > 0:
                theta[i] = theta[i] - np.pi / 2
            else:
                theta[i] = theta[i] + np.pi / 2

    # ensure the range of theta span in (-np.pi / 2, np.pi / 2]
    for i in range(num_samples):
        t = theta[i]
        if t > np.pi / 2:
            t -= np.pi
        elif t <= -np.pi / 2:
            t += np.pi
        theta[i] = t

    # Check whether theta of oriented bounding boxes are within the defined range
    assert torch.logical_and(-np.pi / 2 < theta, theta <= np.pi / 2).all(), \
        ("Theta of oriented bounding boxes are not within the boundary (-pi / 2, pi / 2]")

    return torch.stack((x, y, w, h, theta), -1)


def post_process(predictions, conf_thres=0.5, iou_thres=0.4):
    """
    Args:
        predictions: size-> [batch, ((grid x grid) + (grid x grid) + (grid x grid)) x num_anchors, 8]
                    ex: [1, ((52 x 52) + (26 x 26) + (13 x 13)) x 18, 8] in my case
                    last dimension-> [x, y, w, h, theta, conf, cls_pred]
    Returns:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # Settings
    max_wh = 4096 # min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 5000  # maximum number of boxes into torchvision.ops.nms()
    max_det = 1500

    outputs = [torch.zeros((0, 7), device=predictions.device)] * predictions.size(0)

    for batch, image_pred in enumerate(predictions):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 5] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.shape[0]:
            continue
        # Object confidence times class confidence
        score = image_pred[:, 5] * image_pred[:, 6:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(score).argsort(descending=True)]
        if image_pred.shape[0] > max_nms:
            image_pred = image_pred[:max_nms]
        class_confs, class_preds = image_pred[:, 6:].max(1, keepdim=True)  # class_preds-> index of classes
        dets = torch.cat((image_pred[:, :6], class_confs.float(), class_preds.float()), 1)

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
