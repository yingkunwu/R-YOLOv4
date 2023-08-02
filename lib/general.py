import torch
import numpy as np
import cv2 as cv


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
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y - h / 2
    x3, y3 = x + w / 2, y + h / 2
    x4, y4 = x - w / 2, y + h / 2
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
    w = torch.sqrt(torch.pow((x1 - x2), 2) + torch.pow((y1 - y2), 2))
    h = torch.sqrt(torch.pow((x2 - x3), 2) + torch.pow((y2 - y3), 2))
    theta = -(torch.atan2(y2 - y1, x2 - x1) + torch.atan2(y3 - y4, x3 - x4)) / 2

    # Make the height of bounding boxes always larger then it's width
    for i in range(num_samples):
        if w[i] > h[i]:
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

    return torch.stack((x, y, w, h, theta), -1)