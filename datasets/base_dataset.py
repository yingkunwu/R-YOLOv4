
import glob
import random
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from lib.augmentations import vertical_flip, horisontal_flip, rotate, hsv, gaussian_noise, mixup


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageDataset(Dataset):
    # Reference: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/utils/datasets.py

    def __init__(self, folder_path, img_size=416, ext="png"):
        self.files = sorted(glob.glob(os.path.join(folder_path, "*.{}".format(ext))))
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        #  Hight in {320, 416, 512, 608, ... 320 + 96 * n}
        #  Width in {320, 416, 512, 608, ... 320 + 96 * m}
        img_path = self.files[index % len(self.files)]

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)
        #transform = transforms.ToPILImage(mode="RGB")
        #image = transform(img)
        #image.show()
        return img_path, img

class BaseDataset(Dataset):
    def __init__(self, img_size=608, augment=True, mosaic=True, multiscale=True, normalized_labels=False):
        self.img_files = None
        self.img_size = img_size
        self.augment = augment
        self.mosaic = mosaic
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):
        if self.mosaic:
            img, targets = self.load_mosaic(index)
            if np.random.random() < 0.1:
                img2, targets2 = self.load_mosaic(index)
                img, targets = mixup(img, targets, img2, targets2)
            img = transforms.ToTensor()(img)

        else:
            img, (h, w) = self.load_image(index)
            img = transforms.ToTensor()(img)
            img, pad = pad_to_square(img, 0)

            h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
            _, padded_h, padded_w = img.shape

            targets = self.load_target(index, h_factor, w_factor, pad, padded_h, padded_w)

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = rotate(img, targets)
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
            if np.random.random() < 0.5:
                img, targets = vertical_flip(img, targets)

        return self.img_files[index], img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)

    def load_image(self, index):
        img_path = self.img_files[index]

        # Extract image as PyTorch tensor
        img = np.array(Image.open(img_path).convert('RGB'))
        h, w, c = img.shape
        
        # Handle images with less than three channels
        if c != 3:
            img = np.transpose(np.stack(np.array([img, img, img])), (1, 2, 0))

        if self.augment:
            #img = gaussian_noise(img) # np.random.normal(mean, var ** 0.5, image.shape) would increase run time significantly
            hsv(img)

        return img, (h, w)

    def load_mosaic(self, index):
        """
        Loads 1 image + 3 random images into a 4-image mosaic
        """

        labels4 = []
        s = self.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + [random.randint(0, len(self.img_files) - 1) for _ in range(3)]  # 3 additional image indices
        random.shuffle(indices)

        padded_h, padded_w = s * 2, s * 2

        for i, index in enumerate(indices):
            # Load image
            img, (h, w) = self.load_image(index)
            h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

            # place img in img4
            if i == 0:  # top left
                img4 = np.zeros((padded_h, padded_w, img.shape[2]), dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                padw = xc - x2b
                padh = yc - y2b
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                padw = xc
                padh = yc - y2b
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                padw = xc - x2b
                padh = yc
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                padw = xc
                padh = yc

            img4[y1a:y2a, x1a:x2a, :] = img[y1b:y2b, x1b:x2b, :]  # img4[ymin:ymax, xmin:xmax]

            # Labels
            targets = self.load_target(index, h_factor, w_factor, (padw, padw, padh, padh), padded_h, padded_w, mosaic=True)
            labels4.append(targets)

        # Concat labels
        if len(labels4):
            labels4 = torch.cat(labels4, 0)

        return img4, labels4

    def load_target(self):
        raise NotImplementedError
