import torch
import numpy as np
import os
import tqdm
import glob
from terminaltables import AsciiTable

from tools.options import TestOptions
from tools.post_process import post_process
from tools.utils import load_class_names, skewiou
from tools.load import split_data
from model.model import Yolo

# Reference: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/detect.py

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

        #true_positives.sort()
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


class Test:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = load_class_names(self.args.class_path)
        self.model = None

    def load_model(self):
        model_path = os.path.join("weights", self.args.model_name)
        if os.path.exists(model_path):
            weight_path = glob.glob(os.path.join(model_path, "*.pth"))
            if len(weight_path) == 0:
                assert False, "Model weight not found"
            elif len(weight_path) > 1:
                assert False, "Multiple weights are found. Please keep only one weight in your model directory"
            else:
                weight_path = weight_path[0]
        else:
            assert False, "Model is not exist"
        pretrained_dict = torch.load(weight_path, map_location=torch.device('cpu'))
        self.model = Yolo(n_classes=self.args.number_of_classes)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(pretrained_dict)

    def print_eval_stats(self, metrics_output):
        if metrics_output is not None:
            precision, recall, AP, f1, ap_class = metrics_output
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, self.class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean():.5f} ----")
        else:
            print("---- mAP not measured (no detections found by model) ----")

    def test(self):
        self.load_model()
        self.model.eval()

        # Get dataloader
        test_dataset, test_dataloader = split_data(self.args.data_folder, self.args.img_size, self.args.batch_size,
                                                    shuffle=False, augment=False, multiscale=False)

        print("Compute mAP...")

        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)
        for batch_i, (_, imgs, targets) in enumerate(test_dataloader):
            print(_)
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:6] *= self.args.img_size

            imgs = imgs.to(self.device)

            with torch.no_grad():
                outputs, _ = self.model(imgs)
                outputs = post_process(outputs, conf_thres=self.args.conf_thres, nms_thres=self.args.nms_thres)

            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=self.args.iou_thres)

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        self.print_eval_stats(metrics_output)


if __name__ == "__main__":
    parser = TestOptions()
    args = parser.parse()
    print(args)

    t = Test(args)
    t.test()
