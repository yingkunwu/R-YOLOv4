from .base_dataset import BaseDataset

import os
import numpy as np
import torch

class UCASAODDataset(BaseDataset):
    def load_target(self, index, h_factor, w_factor, pad, padded_h, padded_w, mosaic=False):
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 13))
            label = torch.from_numpy(np.array(self.labels[index]))

            x1, y1, x2, y2, x3, y3, x4, y4 = \
                boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5], boxes[:, 6], boxes[:, 7]

            num_targets = len(boxes)
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