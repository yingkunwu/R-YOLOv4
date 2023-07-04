import torch
import cv2 as cv
import numpy as np
from shapely.geometry import Polygon


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def iou(box1, box2, nms_thres):
    assert len(box1) == 5 and len(box2[0]) == 5

    area1 = box1[2] * box1[3]
    area2 = box2[:, 2] * box2[:, 3]

    box1_ = xywh2xyxy(box1)
    box2_ = xywh2xyxy(box2)

    inter_min_xy = torch.max(box1_[:2], box2_[:, :2])
    inter_max_xy = torch.min(box1_[2:4], box2_[:, 2:4])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    union = area1 + area2 - inter_area

    iou = inter_area / (union + 1e-15)

    mask = iou > 0.3
    large_overlap = torch.zeros(box2.shape[0], dtype=torch.bool)

    large_overlap[mask] = skewiou(box1, box2[mask]) > nms_thres

    return large_overlap


def skewiou(box1, box2):
    assert len(box1) == 5 and len(box2[0]) == 5

    def rbox2polygon(box):
        x, y, w, h, theta = np.array(box)
        points = cv.boxPoints(((x, y), (w, h), theta / np.pi * 180))
        return Polygon(points)

    iou = []
    g = rbox2polygon(box1)
    for i in range(len(box2)):
        p = rbox2polygon(box2[i])

        if not g.is_valid or not p.is_valid:
            raise AssertionError("something went wrong in skew iou")

        inter = g.intersection(p).area
        union = g.area + p.area - inter

        iou.append(torch.tensor(inter / (union + 1e-16)))
    return torch.stack(iou)


def skewiou_2(box1, box2):
    """
    Return skew intersection-over-union of boxes.
    Both sets of boxes are expected to be in (x, y, w, h, theta) format.
    Arguments:
        box1 (Tensor[N, 5])
        box2 (Tensor[M, 5])
    Returns:
        skewiou (Tensor[N, M]): the NxM matrix containing the pairwise
            SkewIoU values for every element in boxes1 and boxes2
    """
    assert len(box1[0]) == 5 and len(box2[0]) == 5

    iou = torch.zeros([box1.shape[0], box2.shape[0]], dtype=torch.float)

    for i, b1 in enumerate(box1):
        iou[i] = skewiou(b1, box2)

    return iou


def post_process(predictions, img_size, conf_thres=0.5, nms_thres=0.4):
    """
    Args:
        predictions: size-> [batch, ((grid x grid) + (grid x grid) + (grid x grid)) x num_anchors, 8]
                    ex: [1, ((52 x 52) + (26 x 26) + (13 x 13)) x 18, 8] in my case
                    last dimension-> [x, y, w, h, a, conf, num_classes]
    Returns:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    batch_size = predictions[0].size(0)
    pred_dim = predictions[0].size(-1) - 1

    predictions_ = []
    for pred in predictions:
        pred[..., :4] = pred[..., :4] * (img_size / pred.size(2))
        pred = torch.cat((pred[..., :5], pred[..., 6:]), -1)
        predictions_.append(pred.view(batch_size, -1, pred_dim))
    predictions = torch.cat(predictions_, 1)
    
    output = [[] for _ in range(batch_size)]

    # Settings
    max_nms = 500 # maximum number of boxes for nms processing
    max_det = 300 # maximum number of detections per image

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
        detections = torch.cat((image_pred[:, :6], class_confs.float(), class_preds.float()), 1).detach().cpu()

        # non-maximum suppression
        keep_boxes = []
        labels = detections[:, -1].unique()
        for label in labels:
            detect = detections[detections[:, -1] == label]
            while len(detect):
                #large_overlap = skewiou(detect[0, :5], detect[:, :5]) > nms_thres
                large_overlap = iou(detect[0, :5], detect[:, :5], nms_thres)
                # Indices of boxes with lower confidence scores, large IOUs and matching labels
                weights = detect[large_overlap, 5:6]
                # Merge overlapping bboxes by order of confidence
                detect[0, :4] = (weights * detect[large_overlap, :4]).sum(0) / weights.sum()
                keep_boxes += [detect[0]]
                detect = detect[~large_overlap]

        if keep_boxes:
            keep_boxes = torch.stack(keep_boxes)
            if keep_boxes.shape[0] > max_det:
                keep_boxes = keep_boxes[:max_det]
            output[batch] = keep_boxes

    return output
