from .base_dataset import BaseDataset

import os
import numpy as np
import torch

class CustomDataset(BaseDataset):
    def load_target(self, index, h_factor, w_factor, pad, padded_h, padded_w, mosaic=False):
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 6))
            num_targets = len(boxes)

            x, y, w, h, theta, label = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5]
            temp_theta = []
            for t in theta:
                if t > np.pi / 2:
                    t = t - np.pi
                elif t <= -(np.pi / 2):
                    t = t + np.pi
                temp_theta.append(t)

            theta = torch.stack(temp_theta)
            assert (-np.pi / 2 < theta).all() or (theta <= np.pi / 2).all()

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

            targets = torch.zeros((len(boxes), 7))
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