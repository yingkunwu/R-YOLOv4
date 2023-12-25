import torch
import numpy as np
import os
import tqdm
import yaml
import argparse
from detectron2.layers.rotated_boxes import pairwise_iou_rotated

from lib.general import post_process
from lib.load import load_data
from lib.loss import ComputeCSLLoss, ComputeKFIoULoss
from lib.logger import logger
from model.yolo import Yolo


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
        v5_metric: Assume maximum recall to be 1.0, as in YOLOv5, MMDetetion etc.
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    # default YOLOv7 metric
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def get_batch_statistics(outputs, targets, iouv, niou):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_stats = []
    for sample_i, pred in enumerate(outputs):
        tar = targets[targets[:, 0] == sample_i, 1:]
        nl = len(tar)
        tcls = tar[:, 0].tolist() if nl else [] # target class

        if len(pred) == 0:
            if nl:
                batch_stats.append((np.zeros((0, niou), dtype=bool), np.empty(0), np.empty(0), tcls))
            continue

        pred_boxes = pred[:, :5]
        pred_scores = pred[:, 5]
        pred_labels = pred[:, 6]

        true_positives = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=targets.device)
        
        if nl:
            detected_boxes = []
            target_labels = tar[:, 0]
            target_boxes = tar[:, 1:6]

            # convert radians to degrees
            pred_boxes[:, 4] = pred_boxes[:, 4] / np.pi * 180
            target_boxes[:, 4] = target_boxes[:, 4] / np.pi * 180

            for cls in torch.unique(target_labels):
                ti = (cls == target_labels).nonzero(as_tuple=False).view(-1)  # target indices
                pi = (cls == pred_labels).nonzero(as_tuple=False).view(-1)  # prediction indices

                if pi.shape[0]:
                    ious, i = pairwise_iou_rotated(pred_boxes[pi], target_boxes[ti]).max(1)

                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]] # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected_boxes.append(d)
                            true_positives[pi[j]] = ious[j] > iouv
                            if len(detected_boxes) == nl: # all targets already located in image
                                break

        # Append statistics (tp, conf, pcls, tcls)
        batch_stats.append((true_positives.cpu(), pred_scores.cpu(), pred_labels.cpu(), tcls))
    return batch_stats


def calculate_eval_stats(stats, num_classes):
    p, r, f1, mp, mr, map50, map = 0., 0., 0., 0., 0., 0., 0.
    ap50, ap, ap_class = [], [], []

    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=num_classes)  # number of targets per class
    else:
        nt = torch.zeros(1)

    return nt, p, r, ap50, ap, f1, ap_class, mp, mr, map50, map


def test(model, compute_loss, device, data, hyp, csl_labels, img_size, batch_size, conf_thres, iou_thres):
    model.eval()

    # Get dataloader
    test_dataset, test_dataloader = load_data(
        data['val'], data['names'], data['type'], hyp, csl_labels, img_size, batch_size, shuffle=False
    )

    logger.info("Compute mAP...")

    stats = []  # List of tuples (tp, conf, pcls, tcls)
    iouv = torch.linspace(0.5, 0.95, 10).to(device) # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    seen = 0
    total_loss_items = {}

    for i, (_, imgs, targets) in enumerate(tqdm.tqdm(test_dataloader)):
        imgs = imgs.to(device)
        targets = targets.to(device)
        seen += len(imgs)

        with torch.no_grad():
            outputs, infer_outputs = model(imgs, training=False)
            _, loss_items = compute_loss(outputs, targets)
            infer_outputs = post_process(infer_outputs, conf_thres=conf_thres, iou_thres=iou_thres)

            for item in loss_items:
                if item in total_loss_items:
                    total_loss_items[item] += loss_items[item]
                else:
                    total_loss_items[item] = loss_items[item]

        # Rescale target
        targets[:, 2:6] *= img_size
        # get sample statistics
        stats += get_batch_statistics(infer_outputs, targets, iouv, niou)

    # Concatenate sample statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]
    # Calculate mAP
    nt, p, r, ap50, ap, f1, ap_class, mp, mr, map50, map = calculate_eval_stats(stats, len(data['names']))

    # Print results
    logger.info(('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95'))

    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    logger.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    for i, c in enumerate(ap_class):
        logger.info(pf % (data['names'][c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # average losses
    for item in total_loss_items:
        total_loss_items[item] /= len(test_dataloader)

    return mp, mr, map50, map, total_loss_items


class Test:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self, n_classes, model_config, mode, ver):
        if not os.path.isfile(self.args.weight_path):
            logger.error("Model weight not found.")
            exit(1)
        pretrained_dict = torch.load(self.args.weight_path, map_location=torch.device('cpu'))
        self.model = Yolo(n_classes, model_config, mode, ver)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(pretrained_dict)

    def run(self):
        # load hyperparameters
        with open(self.args.hyp, "r") as stream:
            config = yaml.safe_load(stream)

        model_cfg, hyp_cfg = config['model'], config['hyp']

        # load data info
        with open(self.args.data, "r") as stream:
            data = yaml.safe_load(stream)

        self.load_model(len(data["names"]), model_cfg, self.args.mode, self.args.ver)

        if self.args.mode == "csl":
            csl = True
            compute_loss = ComputeCSLLoss(self.model, hyp_cfg)
        else:
            csl = False
            compute_loss = ComputeKFIoULoss(self.model, hyp_cfg)

        test(self.model, compute_loss, self.device, data, hyp_cfg, csl,
                self.args.img_size, self.args.batch_size, self.args.conf_thres, self.args.iou_thres)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str, default="", help="file path to load model weight")
    parser.add_argument("--mode", default="csl", nargs='?', choices=['csl', 'kfiou'], help="specify a model type")
    parser.add_argument("--ver", default="yolov5", nargs='?', choices=['yolov4', 'yolov5', 'yolov7'], help="specify a yolo version")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.65, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
    parser.add_argument("--data", type=str, default="", help=".yaml path for data")
    parser.add_argument("--hyp", type=str, default="", help=".yaml path for hyperparameters")

    args = parser.parse_args()
    print(args)

    t = Test(args)
    t.run()
