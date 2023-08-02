import torch
import numpy as np
from detectron2.layers.nms import nms_rotated


def post_process(predictions, conf_thres=0.5, iou_thres=0.4):
    """
    Args:
        predictions: size-> [batch, ((grid x grid) + (grid x grid) + (grid x grid)) x num_anchors, 8]
                    ex: [1, ((52 x 52) + (26 x 26) + (13 x 13)) x 18, 8] in my case
                    last dimension-> [x, y, w, h, a, conf, num_classes]
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
