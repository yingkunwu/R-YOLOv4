import torch
import argparse
import numpy as np
from tools.plot import load_class_names
from tools.post_process import post_process
from tools.utils import get_batch_statistics, ap_per_class
from tools.load import split_data
from model.model import Yolo

# Reference: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/detect.py


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_folder", type=str, default="data/test3", help="path to dataset")
    parser.add_argument("--weights_path", type=str, default="weights/AOD_800.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.7, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold for evaluation")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    class_names = load_class_names(args.class_path)
    pretrained_dict = torch.load(args.weights_path, map_location=torch.device('cpu'))
    model = Yolo(n_classes=2)
    model = model.to(device)
    model.load_state_dict(pretrained_dict)

    model.eval()

    # Get dataloader
    train_dataset, train_dataloader = split_data(args.test_folder, args.img_size, args.batch_size,
                                                 shuffle=False, augment=False, multiscale=False)

    print("Compute mAP...")

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(train_dataloader):
        print(_)
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:6] *= args.img_size

        imgs = torch.autograd.Variable(imgs.type(FloatTensor), requires_grad=False)

        with torch.no_grad():
            outputs, _ = model(imgs)
            outputs = post_process(outputs, conf_thres=args.conf_thres, nms_thres=args.nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=args.iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
