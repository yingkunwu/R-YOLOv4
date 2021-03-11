import numpy as np
import torch
import tqdm
from shapely.geometry import Polygon


def R(theta):
    """
    Args:
        theta: must be radian
    Returns: rotation matrix
    """
    r = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]])
    return r


def T(x, y):
    """
    Args:
        x, y: values to translate
    Returns: translation matrix
    """
    t = torch.tensor([[1, 0, x], [0, 1, y], [0, 0, 1]])
    return t


def rotate(center_x, center_y, a, p):
    P = torch.matmul(T(center_x, center_y), torch.matmul(R(a), torch.matmul(T(-center_x, -center_y), p)))
    return P[:2]


def xywha2xyxyxyxy(p):
    """
    Args:
        p: 1-d tensor which contains (x, y, w, h, a)
    Returns: bbox coordinates (x1, y1, x2, y2, x3, y3, x4, y4) which is transferred from (x, y, w, h, a)
    """
    x, y, w, h, a = p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4]

    x1, y1, x2, y2 = x + w / 2, y - h / 2, x + w / 2, y + h / 2
    x3, y3, x4, y4 = x - w / 2, y + h / 2, x - w / 2, y - h / 2

    P1 = torch.tensor((x1, y1, 1)).reshape(3, -1)
    P2 = torch.tensor((x2, y2, 1)).reshape(3, -1)
    P3 = torch.tensor((x3, y3, 1)).reshape(3, -1)
    P4 = torch.tensor((x4, y4, 1)).reshape(3, -1)
    P = torch.stack((P1, P2, P3, P4)).squeeze(2).T
    P = rotate(x, y, a, P)
    X1, X2, X3, X4 = P[0]
    Y1, Y2, Y3, Y4 = P[1]

    return X1, Y1, X2, Y2, X3, Y3, X4, Y4


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def skewiou(box1, box2):
    assert len(box1) == 5 and len(box2[0]) == 5
    iou = []
    g = torch.stack(xywha2xyxyxyxy(box1))
    g = Polygon(g.reshape((4, 2)))
    for i in range(len(box2)):
        p = torch.stack(xywha2xyxyxyxy(box2[i]))
        p = Polygon(p.reshape((4, 2)))
        if not g.is_valid or not p.is_valid:
            print("something went wrong in skew iou")
            return 0
        inter = Polygon(g).intersection(Polygon(p)).area
        union = g.area + p.area - inter
        iou.append(torch.tensor(inter / (union + 1e-16)))
    return torch.stack(iou)


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :5]
        pred_scores = output[:, 5]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = skewiou(pred_box, target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        true_positives.sort()
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics
