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
    def __init__(self, img_size=608, sample_size=600, augment=True, mosaic=True, multiscale=True, normalized_labels=False):
        self.img_size = img_size
        self.augment = augment
        self.mosaic = mosaic
        self.mosaic_sample_size = sample_size
        self.mosaic_border = [-self.mosaic_sample_size // 2, -self.mosaic_sample_size // 2]
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

        self.img_files = None
        self.label_files = None
        self.category = None

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

            label_factor = (h, w) if self.normalized_labels else (1, 1)
            padded_size = img.shape[1:]
            boundary = (0, w, 0, h)

            targets = self.load_target(index, label_factor, pad, padded_size, boundary)

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
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
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
        Loads 1 image + 3 random images into a 4-image mosaic.
        Each image is cropped based on the sameple_size.
        A larger sample size means more information in each image would be used.
        """

        labels4 = []
        s = self.mosaic_sample_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + [random.randint(0, len(self.img_files) - 1) for _ in range(3)]  # 3 additional image indices
        random.shuffle(indices)

        h_padded, w_padded = s * 2, s * 2
        padded_size = (h_padded, w_padded)

        for i, index in enumerate(indices):
            img, (h, w) = self.load_image(index)
            label_factor = (h, w) if self.normalized_labels else (1, 1)

            # place img in img4
            if i == 0:  # top left
                img4 = np.zeros((h_padded, w_padded, img.shape[2]), dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                xt, yt = int(random.uniform((x2a - x1a), w)), int(random.uniform((y2a - y1a), h))
                x1b, y1b, x2b, y2b = xt - (x2a - x1a), yt - (y2a - y1a), xt, yt  # xmin, ymin, xmax, ymax (small image)
                padw = xc - x2b
                padh = yc - y2b
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                xt, yt = int(random.uniform((x2a - x1a), w)), int(random.uniform((y2a - y1a), h))
                x1b, y1b, x2b, y2b = xt - (x2a - x1a), yt - (y2a - y1a), xt, yt
                padw = xc - x1b
                padh = yc - y2b
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                xt, yt = int(random.uniform((x2a - x1a), w)), int(random.uniform((y2a - y1a), h))
                x1b, y1b, x2b, y2b = xt - (x2a - x1a), yt - (y2a - y1a), xt, yt
                padw = xc - x2b
                padh = yc - y1b
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                xt, yt = int(random.uniform((x2a - x1a), w)), int(random.uniform((y2a - y1a), h))
                x1b, y1b, x2b, y2b = xt - (x2a - x1a), yt - (y2a - y1a), xt, yt
                padw = xc - x1b
                padh = yc - y1b

            img4[y1a:y2a, x1a:x2a, :] = img[y1b:y2b, x1b:x2b, :]  # img4[ymin:ymax, xmin:xmax]

            # Labels
            pad = (padw, padw, padh, padh)
            boundary = (x1b, x2b, y1b, y2b)
            targets = self.load_target(index, label_factor, pad, padded_size, boundary)
            labels4.append(targets)

        # Concat labels
        if len(labels4):
            labels4 = torch.cat(labels4, 0)

        return img4, labels4


    def load_target(self, index, label_factor, pad, padded_size, boundary):
        """
        Args:
            index: index of label files going to be load
            label_factor: factor that resize labels to the same size with images
            pad: the amount of zero pixel value that are padded beside images
            padded_size: the size of images after padding
            boundary: the boundary of targets

        Returns:
            Normalized labels of objects -> [batch_index, label, x, y, w, h, theta] -> torch.Size([num_targets, 7])
        """
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        if os.path.exists(label_path):
            x, y, w, h, theta, label, num_targets = self.load_files(label_path)

            # Return zero length tersor if there is no object in the image
            if not num_targets:
                return torch.zeros((0, 7))

            # Check whether theta of oriented bounding boxes are within the boundary or not
            if not ((-np.pi / 2 < theta).all() or (theta <= np.pi / 2).all()):
                raise AssertionError("Theta of oriented bounding boxes are not within the boundary (-pi / 2, pi / 2]")

            # Make the height of bounding boxes always larger then it's width
            for i in range(num_targets):
                if w[i] > h[i]:
                    temp1, temp2 = h[i].clone(), w[i].clone()
                    w[i], h[i] = temp1, temp2
                    if theta[i] > 0:
                        theta[i] = theta[i] - np.pi / 2
                    else:
                        theta[i] = theta[i] + np.pi / 2

            # Make the scale of coordinates to the same size of images
            h_factor, w_factor = label_factor
            x *= w_factor
            y *= h_factor
            w *= w_factor
            h *= h_factor

            # Remove objects that exceed the size of images or the cropped area when doing mosaic augmentation
            left_boundary, right_boundary, top_boundary, bottom_boundary = boundary
            mask = torch.ones_like(x)
            mask = torch.logical_and(mask, x > left_boundary)
            mask = torch.logical_and(mask, x < right_boundary)
            mask = torch.logical_and(mask, y > top_boundary)
            mask = torch.logical_and(mask, y < bottom_boundary)

            label = label[mask]
            x = x[mask]
            y = y[mask]
            w = w[mask]
            h = h[mask]
            theta = theta[mask]

            # Relocalize coordinates based on images padding or mosaic augmentation
            x1 = (x - w / 2) + pad[0]
            y1 = (y - h / 2) + pad[2]
            x2 = (x + w / 2) + pad[1]
            y2 = (y + h / 2) + pad[3]

            # Normalized coordinates
            padded_h, padded_w = padded_size
            x = ((x1 + x2) / 2) / padded_w
            y = ((y1 + y2) / 2) / padded_h
            w /= padded_w
            h /= padded_h

            targets = torch.zeros((len(label), 7))
            targets[:, 1] = label
            targets[:, 2] = x
            targets[:, 3] = y
            targets[:, 4] = w
            targets[:, 5] = h
            targets[:, 6] = theta
            return targets

        else:
            print(label_path)
            assert False, "Label file not found"


    def load_files(self):
        raise NotImplementedError
