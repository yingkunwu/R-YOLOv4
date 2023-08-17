import glob
import random
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from lib.augmentations import hsv, vertical_flip, horizontal_flip, mixup, random_warping
from lib.general import xyxyxyxy2xywha


def gaussian_label(label, num_class, u=0, sig=4.0):
    """
    ref: https://github.com/hukaixuan19970627/yolov5_obb/blob/master/utils/rboxs_utils.py

    Convert to CSL Labels:
        CSL map the class of each angle into gaussian distribution to handle the periodicity of the angle 
        and increase the error tolerance to adjacent angles
    Args:
        label (float32):[1], theta class
        num_theta_class (int): [1], theta class num
        u (float32):[1], mean in gaussian function
        sig (float32):[1], std in gaussian function, which is window radius for Circular Smooth Label
    Returns:
        csl_label (array): [num_theta_class], gaussian function smooth label
    """
    x = np.arange(-num_class/2, num_class/2)
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2))
    index = int(num_class/2 - label)
    return np.concatenate([y_sig[index:], y_sig[:index]], axis=0)


def pad_to_square(img, new_shape, pad_value):
    shape = img.shape[:2]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)  # add border

    return img, (dh, dw)


class ImageDataset(Dataset):
    def __init__(self, folder_path, img_size=416, ext="png"):
        self.files = sorted(glob.glob(os.path.join(folder_path, "*.{}".format(ext))))
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        #  Hight in {320, 416, 512, 608, ... 320 + 96 * n}
        #  Width in {320, 416, 512, 608, ... 320 + 96 * m}
        img_path = self.files[index % len(self.files)]

        # Read image from file
        img = cv2.imread(img_path)
        # Pad to square resolution
        img, _ = pad_to_square(img, (self.img_size, self.img_size), (114, 114, 114))
        # Turn into tensor
        img = img.transpose((2, 0, 1))[::-1] # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255

        return img_path, img

class BaseDataset(Dataset):
    def __init__(self, hyp, img_size, augment, csl, normalized_labels):
        self.hyp = hyp
        self.img_size = img_size
        self.augment = augment
        self.normalized_labels = normalized_labels
        self.csl = csl
        self.mosaic_border = [-img_size // 2, -img_size // 2]

    def __getitem__(self, index):
        if self.augment and random.random() < self.hyp['mosaic']:
            # mosaic augmentation
            if random.random() < 0.8:
                img, targets = self.load_mosaic(index)
            else:
                img, targets = self.load_mosaic9(index)
            # perform rotate, scale, and translate augmentations
            img, targets = random_warping(
                img, targets, self.hyp['rotate'], self.hyp['scale'], self.hyp['translate'], self.mosaic_border
            )
            # mixup augmentation
            if np.random.random() < self.hyp['mixup']:
                if random.random() < 0.8:
                    img2, targets2 = self.load_mosaic(random.randint(0, len(self.img_files) - 1))
                else:
                    img2, targets2 = self.load_mosaic9(random.randint(0, len(self.img_files) - 1))
                # perform rotate, scale, and translate augmentations
                img2, targets2 = random_warping(
                    img2, targets2, self.hyp['rotate'], self.hyp['scale'], self.hyp['translate'], self.mosaic_border
                )
                img, targets = mixup(img, targets, img2, targets2)
        else:
            img, (h0, w0), (h, w) = self.load_image(index)
            img, pad = pad_to_square(img, (self.img_size, self.img_size), (114, 114, 114))

            targets = self.load_target(index, pad, (h0, w0), (h, w))

            if self.augment: 
                # perform rotate, scale, and translate augmentations
                img, targets = random_warping(img, targets, self.hyp['rotate'], self.hyp['scale'], self.hyp['translate'])

        # Remove objects that exceed the size of images
        targets = self.filtering(targets, (0, img.shape[1], 0, img.shape[0]))
        targets = self.normalize(targets, img.shape[:2])

        # horizontal flip augmentation
        if self.augment and np.random.random() < self.hyp['fliplr']:
            img, targets = horizontal_flip(img, targets)
        # vertical flip augmentation
        if self.augment and np.random.random() < self.hyp['flipud']:
            img, targets = vertical_flip(img, targets)

        # prepare empty labels if the number of targets are 0
        labels = torch.zeros((0, 187), dtype=torch.float32) if self.csl else torch.zeros((0, 7), dtype=torch.float32)

        if len(targets):
            # convert poly bboxes to oriented bboxes
            rboxes = xyxyxyxy2xywha(targets[:, 2:])

            if self.csl:
                csl_labels = []
                for i in range(len(rboxes)):
                    angle = rboxes[i, 4] * 180 / np.pi + 90
                    csl_label = gaussian_label(label=angle, num_class=180, u=0, sig=6)
                    csl_labels.append(csl_label)
                csl_labels = torch.from_numpy(np.stack(csl_labels)).type(torch.float32)

                labels = torch.cat((targets[:, :2], rboxes, csl_labels), -1)
            else:
                labels = torch.cat((targets[:, :2], rboxes), -1)

        # Convert
        img = img.transpose((2, 0, 1))[::-1] # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255

        return self.img_files[index], img, labels

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        return paths, torch.stack(imgs, 0), torch.cat(targets, 0)

    def __len__(self):
        return len(self.img_files)

    def load_image(self, index):
        img_path = self.img_files[index]

        # Extract image as PyTorch tensor
        img = cv2.imread(img_path)
        h, w, c = img.shape
        
        # Handle images with less than three channels
        if c != 3:
            img = np.transpose(np.stack(np.array([img, img, img])), (1, 2, 0))

        r = self.img_size / max(h, w)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=interp)

        if self.augment:
            hsv(img, self.hyp['hsv_h'], self.hyp['hsv_s'], self.hyp['hsv_v'])

        return img, (h, w), img.shape[:2]

    def load_target(self, index, pad, img_size0, img_size, boarder=None):
        """
        Args:
            index: index of label files going to be load
            pad: the amount of zero pixel value that are padded beside images
            img_size0: original image size
            img_size: image size after resizing
            boarder: exceed which targets should be discard
        Returns:
            Normalized labels of objects -> [batch_index, label, x, y, w, h, theta] -> torch.Size([num_targets, 7])
        """
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        if os.path.exists(label_path):
            polys, labels = self.load_files(label_path)

            # Return zero length tersor if there is no object in the image
            if not len(labels):
                return torch.zeros((0, 10))

            # Normalizd coordinates if it has not been normalized yet
            if not self.normalized_labels:
                h0, w0 = img_size0
                polys[:, [0, 2, 4, 6]] /= w0
                polys[:, [1, 3, 5, 7]] /= h0

            # Rescale the scale of coordinates to the same scale of self.img_size
            h_, w_ = img_size
            polys[:, [0, 2, 4, 6]] *= w_
            polys[:, [1, 3, 5, 7]] *= h_

            # Create targets
            targets = torch.zeros((len(labels), 10))
            targets[:, 1:] = torch.cat((labels.unsqueeze(-1), polys), -1)

            if boarder is not None:
                targets = self.filtering(targets, boarder)

            # Relocalize coordinates based on images padding
            targets[:, [2, 4, 6, 8]] += pad[1]
            targets[:, [3, 5, 7, 9]] += pad[0]

            return targets

        else:
            assert False, "Label file {} not found".format(label_path)

    def load_mosaic(self, index):
        # loads 1 image + 3 random images into a 4-image mosaic

        labels4 = []
        s = self.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border] # mosaic center x, y
        indices = [index] + random.choices(range(len(self.img_files)), k=3) # 3 additional image indices

        for i, index in enumerate(indices):
            img, (h0, w0), (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            pad = (padh, padw)
            targets = self.load_target(index, pad, (h0, w0), (h, w), boarder=(x1b, x2b, y1b, y2b))
            labels4.append(targets)

        # Concat labels
        labels4 = torch.cat(labels4, 0)

        return img4, labels4
    
    def load_mosaic9(self, index):
        # loads images in a 9-mosaic

        labels9 = []
        s = self.img_size
        indices = [index] + random.choices(range(len(self.img_files)), k=8) # 8 additional image indices

        for i, index in enumerate(indices):
            img, (h0, w0), (h, w) = self.load_image(index)

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h_, w_ = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w_, s, s + w_ + w, s + h
            elif i == 4:  # bottom right
                c = s + w_, s + hp, s + w_ + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w_ - w, s + h_, s + w_, s + h_ + h
            elif i == 6:  # bottom left
                c = s + w_ - wp - w, s + h_, s + w_ - wp, s + h_ + h
            elif i == 7:  # left
                c = s - w, s + h_ - h, s, s + h_
            elif i == 8:  # top left
                c = s - w, s + h_ - hp - h, s, s + h_ - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

            # Labels
            pad = (pady, padx)
            targets = self.load_target(index, pad, (h0, w0), (h, w), boarder=(x1 - padx, w, y1 - pady, h))
            labels9.append(targets)

        # Concat labels
        labels9 = torch.cat(labels9, 0)

        # Offset
        yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        # Remove objects that exceed the size of images (the cropped area) when doing mosaic augmentation
        labels9 = self.filtering(labels9, (xc, xc + 2 * s, yc, yc + 2 * s))
        
        # Translate labels
        labels9[:, [2, 4, 6, 8]] -= xc
        labels9[:, [3, 5, 7, 9]] -= yc

        return img9, labels9

    @staticmethod
    def filtering(targets, boarder):
        # Remove objects that exceed the specified region
        x1, x2, y1, y2 = boarder

        x = torch.mean(targets[:, [2, 4, 6, 8]], dim=1)
        y = torch.mean(targets[:, [3, 5, 7, 9]], dim=1)

        mask = (
            (x > x1) & (x < x2) & (y > y1) & (y < y2)
        )

        return targets[mask]
    
    @staticmethod
    def normalize(targets, img_size):
        # Normalize x, y, w, h, of targets into [0, 1]
        height, width = img_size
        targets[:, [2, 4, 6, 8]] /= width
        targets[:, [3, 5, 7, 9]] /= height

        return targets
    
    def load_files(self):
        raise NotImplementedError
