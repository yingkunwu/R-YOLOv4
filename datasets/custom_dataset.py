import os
import numpy as np
import torch
import glob
import warnings

from .base_dataset import BaseDataset

class CustomDataset(BaseDataset):
    def __init__(self, data_dir, img_size=416, augment=True, mosaic=True, multiscale=True, normalized_labels=False):
        super().__init__(img_size, augment, mosaic, multiscale, normalized_labels)
        self.img_files = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.label_files = [path.replace(".jpg", ".txt") for path in self.img_files]

    def load_files(self, label_path):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 6))
        num_targets = len(boxes)

        if not num_targets:
            return None, None, None, None, None, None, 0

        x, y, w, h, theta, label = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5]
        temp_theta = []
        for t in theta:
            if t > np.pi / 2:
                t = t - np.pi
            elif t <= -(np.pi / 2):
                t = t + np.pi
            temp_theta.append(t)

        theta = torch.stack(temp_theta)

        return x, y, w, h, theta, label, num_targets
