import os
import cv2
import numpy as np
import torch
import glob

from lib.general import xyxyxyxy2xywha
from .base_dataset import BaseDataset


class UCASAODDataset(BaseDataset):
    def __init__(self, data_dir, class_names, hyp, augment=False, img_size=416, normalized_labels=False):
        super().__init__(hyp, img_size, augment, normalized_labels)
        self.img_files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        self.label_files = [path.replace(".png", ".txt") for path in self.img_files]
        self.category = {}
        for i, name in enumerate(class_names):
            self.category[name.replace(" ", "-")] = i

    def load_files(self, label_path):
        lines = open(label_path, 'r').readlines()

        x1, y1, x2, y2, x3, y3, x4, y4, labels = [], [], [], [], [], [], [], [], []
        for line in lines:
            line = line.split('\t')
            x1.append(float(line[1]))
            y1.append(float(line[2]))
            x2.append(float(line[3]))
            y2.append(float(line[4]))
            x3.append(float(line[5]))
            y3.append(float(line[6]))
            x4.append(float(line[7]))
            y4.append(float(line[8]))
            labels.append(self.category[line[0]])

        if len(labels):
            x1 = torch.tensor(x1)
            y1 = torch.tensor(y1)
            x2 = torch.tensor(x2)
            y2 = torch.tensor(y2)
            x3 = torch.tensor(x3)
            y3 = torch.tensor(y3)
            x4 = torch.tensor(x4)
            y4 = torch.tensor(y4)
            labels = torch.tensor(labels)

            boxes = torch.stack((x1, y1, x2, y2, x3, y3, x4, y4), -1)
            rboxes = xyxyxyxy2xywha(boxes)

            return rboxes, labels

        else:
            return None, None, None, None, None, None, 0