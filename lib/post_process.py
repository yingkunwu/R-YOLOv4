import torch
import cv2 as cv
import numpy as np
from shapely.geometry import Polygon


def rbox2polygon(box):
    x, y, w, h, theta = np.array(box)
    points = cv.boxPoints(((x, y), (w, h), theta / np.pi * 180))
    return Polygon(points)


def skewiou(box1, box2):
    assert len(box1) == 5 and len(box2[0]) == 5
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


def post_process(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Args:
        prediction: size-> [batch, ((grid x grid) + (grid x grid) + (grid x grid)) x num_anchors, 8]
                    ex: [1, ((52 x 52) + (26 x 26) + (13 x 13)) x 18, 8] in my case
                    last dimension-> [x, y, w, h, a, conf, num_classes]
    Returns:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    output = [None for _ in range(len(prediction))]
    for batch, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 5] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 5] * image_pred[:, 6:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 6:].max(1, keepdim=True)  # class_preds-> index of classes
        detections = torch.cat((image_pred[:, :6], class_confs.float(), class_preds.float()), 1).detach().cpu()

        # non-maximum suppression
        keep_boxes = []
        labels = detections[:, -1].unique()
        for label in labels:
            detect = detections[detections[:, -1] == label]
            while len(detect):
                large_overlap = skewiou(detect[0, :5], detect[:, :5].detach()) > nms_thres
                # Indices of boxes with lower confidence scores, large IOUs and matching labels
                weights = detect[large_overlap, 5:6]
                # Merge overlapping bboxes by order of confidence
                detect[0, :4] = (weights * detect[large_overlap, :4]).sum(0) / weights.sum()
                keep_boxes += [detect[0]]
                detect = detect[~large_overlap]
            if keep_boxes:
                output[batch] = torch.stack(keep_boxes)

    return output
