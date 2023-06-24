import torch
import numpy as np
import os
import tqdm
import glob
from terminaltables import AsciiTable

from lib.options import TestOptions
from lib.post_process import post_process, skewiou_2
from lib.load import load_data
from lib.utils import load_class_names
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
        pred_labels = pred[:, -1]

        true_positives = np.zeros((pred.shape[0], niou), dtype=bool)

        if nl:
            detected_boxes = []
            target_labels = tar[:, 0]
            target_boxes = tar[:, 1:]

            for cls in np.unique(target_labels):
                ti = np.nonzero(cls == target_labels).flatten()
                pi = np.nonzero(cls == pred_labels).flatten()

                if pi.shape[0]:
                    ious, i = skewiou_2(pred_boxes[pi], target_boxes[ti]).max(1) # best ious, indices

                    ious = ious.numpy()

                    detected_set = set()
                    for j in np.nonzero(ious > iouv[0])[0]:
                        d = ti[i[j]] # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected_boxes.append(d)
                            true_positives[pi[j]] = ious[j] > iouv
                            if len(detected_boxes) == nl: # all targets already located in image
                                break

        # Append statistics (tp, conf, pcls, tcls)
        batch_stats.append((true_positives, pred_scores, pred_labels, tcls))
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


def test(model, device, data_folder, dataset, img_size, batch_size, conf_thres, nms_thres, class_names):
    model.eval()

    # Get dataloader
    test_dataset, test_dataloader = load_data(data_folder, dataset, "test", img_size, batch_size=batch_size, 
                                                shuffle=False, augment=False, mosaic=False, multiscale=False)

    print("Compute mAP...")

    stats = []  # List of tuples (tp, conf, pcls, tcls)
    iouv = np.linspace(0.5, 0.95, 10) # iou vector for mAP@0.5:0.95
    niou = np.size(iouv)
    seen = 0

    for i, (_, imgs, targets) in enumerate(tqdm.tqdm(test_dataloader)):
        imgs = imgs.to(device)
        seen += 1

        with torch.no_grad():
            outputs, loss = model(imgs)
            outputs = post_process(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        # Rescale target
        targets[:, 2:6] *= img_size
        # get sample statistics
        stats += get_batch_statistics(outputs, targets, iouv, niou)

    # Concatenate sample statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]
    # Calculate mAP
    nt, p, r, ap50, ap, f1, ap_class, mp, mr, map50, map = calculate_eval_stats(stats, len(class_names))

    # Print results
    print(('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95'))

    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    for i, c in enumerate(ap_class):
        print(pf % (class_names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    return mp, mr, map50, map


class Test:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = load_class_names(os.path.join(self.args.data_folder, "class.names"))
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

    def run(self):
        self.load_model()

        test(self.model, self.device, self.args.data_folder, self.args.dataset, self.args.img_size, self.args.batch_size, 
                self.args.conf_thres, self.args.nms_thres, self.class_names)



if __name__ == "__main__":
    parser = TestOptions()
    args = parser.parse()
    print(args)

    t = Test(args)
    t.run()
