import os
import numpy as np
import torch
import glob

from .base_dataset import BaseDataset

class DOTADataset(BaseDataset):
    def __init__(self, data_dir, class_names, hyp, augment, img_size, csl, normalized_labels=False):
        super().__init__(hyp, img_size, augment, csl, normalized_labels)
        self.img_files = sorted(glob.glob(os.path.join(data_dir, "images", "*.png")))
        self.label_files = [path.replace("images", "annfiles").replace(".png", ".txt") for path in self.img_files]
        self.category = {}
        for i, name in enumerate(class_names):
            self.category[name.replace(" ", "-")] = i

    def load_files(self, label_path):
        lines = open(label_path, 'r').readlines()

        x1, y1, x2, y2, x3, y3, x4, y4, labels = [], [], [], [], [], [], [], [], []

        for line in lines:
            line = line.split(' ')
            x1.append(float(line[0]))
            y1.append(float(line[1]))
            x2.append(float(line[2]))
            y2.append(float(line[3]))
            x3.append(float(line[4]))
            y3.append(float(line[5]))
            x4.append(float(line[6]))
            y4.append(float(line[7]))
            labels.append(self.category[line[8]])

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

            polys = torch.stack((x1, y1, x2, y2, x3, y3, x4, y4), -1).type(torch.float32)

        else:
            polys = torch.zeros((0, 8), dtype=torch.float32)

        return polys, labels