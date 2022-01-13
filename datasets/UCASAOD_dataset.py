import os
import numpy as np
import torch
import glob

from .base_dataset import BaseDataset
from lib.utils import load_class_names

class UCASAODDataset(BaseDataset):
    def __init__(self, data_dir, class_names, img_size=416, augment=True, mosaic=True, multiscale=True, normalized_labels=False):
        super().__init__(img_size, augment, mosaic, multiscale, normalized_labels)
        self.img_files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        self.label_files = [path.replace(".png", ".txt") for path in self.img_files]
        self.category = {}
        for i, name in enumerate(class_names):
            self.category[name.replace(" ", "-")] = i

    def load_target(self, index, h_factor, w_factor, pad, padded_h, padded_w, mosaic=False):
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        if os.path.exists(label_path):
            lines = open(label_path, 'r').readlines()

            x1, y1, x2, y2, x3, y3, x4, y4, label = [], [], [], [], [], [], [], [], []
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
                label.append(self.category[line[0]])

            x1 = torch.tensor(x1)
            y1 = torch.tensor(y1)
            x2 = torch.tensor(x2)
            y2 = torch.tensor(y2)
            x3 = torch.tensor(x3)
            y3 = torch.tensor(y3)
            x4 = torch.tensor(x4)
            y4 = torch.tensor(y4)
            label = torch.tensor(label)

            num_targets = len(label)
            x = ((x1 + x3) / 2 + (x2 + x4) / 2) / 2
            y = ((y1 + y3) / 2 + (y2 + y4) / 2) / 2
            w = torch.sqrt(torch.pow((x1 - x2), 2) + torch.pow((y1 - y2), 2))
            h = torch.sqrt(torch.pow((x2 - x3), 2) + torch.pow((y2 - y3), 2))

            theta = ((y2 - y1) / (x2 - x1 + 1e-16) + (y3 - y4) / (x3 - x4 + 1e-16)) / 2
            theta = torch.atan(theta)
            theta = torch.stack([t if t != -(np.pi / 2) else t + np.pi for t in theta])
            for t in theta:
                assert -(np.pi / 2) < t <= (np.pi / 2), "angle: " + str(t)

            for i in range(num_targets):
                if w[i] < h[i]:
                    temp1, temp2 = h[i].clone(), w[i].clone()
                    w[i], h[i] = temp1, temp2
                    if theta[i] > 0:
                        theta[i] = theta[i] - np.pi / 2
                    else:
                        theta[i] = theta[i] + np.pi / 2
            assert (-np.pi / 2 < theta).all() or (theta <= np.pi / 2).all()

            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (x - w / 2)
            y1 = h_factor * (y - h / 2)
            x2 = w_factor * (x + w / 2)
            y2 = h_factor * (y + h / 2)

            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            # Returns (x, y, w, h)
            x = ((x1 + x2) / 2) / padded_w
            y = ((y1 + y2) / 2) / padded_h
            w *= w_factor / padded_w
            h *= h_factor / padded_h

            targets = torch.zeros((len(label), 7))
            targets[:, 1] = label
            targets[:, 2] = x
            targets[:, 3] = y
            targets[:, 4] = w
            targets[:, 5] = h
            targets[:, 6] = theta

            if mosaic:
                mask = torch.ones_like(targets[:, 1])
                mask = torch.logical_and(mask, targets[:, 2] > 0)
                mask = torch.logical_and(mask, targets[:, 2] < 1)
                mask = torch.logical_and(mask, targets[:, 3] > 0)
                mask = torch.logical_and(mask, targets[:, 3] < 1)

                return targets[mask]
            else:
                return targets

        else:
            print(label_path)
            assert False, "Label file not found"
