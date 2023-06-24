import glob
import random
import os
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from lib.augmentations import vertical_flip, horisontal_flip, hsv, gaussian_noise, mixup, random_warping


def pad_to_square(img, pad_value):
    h, w, c = img.shape
    # (upper / left) padding
    pad = np.abs(h - w) // 2
    # Determine padding
    if h <= w:
        padw = 0
        padh = pad
    else:
        padw = pad
        padh = 0

    top, bottom = int(round(padh - 0.1)), int(round(padh + 0.1))
    left, right = int(round(padw - 0.1)), int(round(padw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)  # add border

    return img, (padh, padw)


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageDataset(Dataset):
    # TODO: apply ImageDataset to new functions
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
        self.img_size = img_size
        self.augment = augment
        self.mosaic = mosaic
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

        self.img_files = None
        self.label_files = None

    def __getitem__(self, index):
        if self.mosaic:
            if random.random() < 0.8:
                img, targets = self.load_mosaic(index)
            else:
                img, targets = self.load_mosaic9(index)
            img, targets = random_warping(img, targets, scale = .5, translate = .1)

            if np.random.random() < 0.5:
                if random.random() < 0.8:
                    img2, targets2 = self.load_mosaic(random.randint(0, len(self.img_files) - 1))
                else:
                    img2, targets2 = self.load_mosaic9(random.randint(0, len(self.img_files) - 1))
                img, targets = random_warping(img, targets, scale = .5, translate = .1)

                img, targets = mixup(img, targets, img2, targets2)
        else:
            img, (h0, w0), (h, w) = self.load_image(index)
            img, pad = pad_to_square(img, 0)

            targets = self.load_target(index, pad, (h0, w0), (h, w), img.shape[:2])
            img, targets = random_warping(img, targets, scale = .5, translate = .1)

        targets = self.filtering(targets, img.shape[:2])
        targets = self.normalize(targets, img.shape[:2])
        img = transforms.ToTensor()(img)

        # TODO: reshpae image size to img_size

        # Apply augmentations
        if self.augment:
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
        # TODO: make sure we don't need this
        # Selects new image size every tenth batch
        #if self.multiscale and self.batch_count % 10 == 0:
        #    self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        #imgs = torch.stack(imgs)
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

        r = self.img_size / max(h, w)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=interp)

        if self.augment:
            #img = gaussian_noise(img) # np.random.normal(mean, var ** 0.5, image.shape) would increase run time significantly
            hsv(img)

        return img, (h, w), img.shape[:2]

    def load_target(self, index, pad, img_size0, img_size, padded_size):
        """
        Args:
            index: index of label files going to be load
            label_factor: factor that resize labels to the same size with images
            pad: the amount of zero pixel value that are padded beside images
            padded_size: the size of images after padding

        Returns:
            Normalized labels of objects -> [batch_index, label, x, y, w, h, theta] -> torch.Size([num_targets, 7])
        """
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        if os.path.exists(label_path):
            x, y, w, h, theta, label, num_targets = self.load_files(label_path)

            # Return zero length tersor if there is no object in the image
            if not num_targets:
                return torch.zeros((0, 7))

            # Check whether theta of oriented bounding boxes are within the defined range or not
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

            # Normalizd coordinates if it has not been normalized yet
            if not self.normalized_labels:
                h0, w0 = img_size0
                x /= w0
                y /= h0
                w /= w0
                h /= h0

            # Rescale the scale of coordinates to the same scale of self.img_size
            h_, w_ = img_size
            x *= w_
            y *= h_
            w *= w_
            h *= h_

            # Relocalize coordinates based on images padding
            x += pad[1]
            y += pad[0]

            # Create targets
            targets = torch.zeros((len(label), 7))
            targets[:, 1:] = torch.vstack([label, x, y, w, h, theta]).T

            # Remove objects that exceed the size of images after padding
            targets = self.filtering(targets, padded_size)

            return targets

        else:
            print(label_path)
            assert False, "Label file not found"

    def load_mosaic(self, index):
        """
        Loads 1 image + 3 random images into a 4-image mosaic.
        Each image is cropped based on the sameple_size.
        A larger sample size means more information in each image would be used.
        """

        labels4 = []
        s = self.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border] # mosaic center x, y
        indices = [index] + random.choices(range(len(self.img_files)), k=3) # 3 additional image indices

        padded_size = (s * 2, s * 2)

        for i, index in enumerate(indices):
            img, (h0, w0), (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.zeros((s * 2, s * 2, img.shape[2]), dtype=np.uint8)
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
            targets = self.load_target(index, pad, (h0, w0), (h, w), padded_size)
            labels4.append(targets)

        # Concat labels
        labels4 = torch.cat(labels4, 0)

        return img4, labels4
    
    def load_mosaic9(self, index):
        # loads images in a 9-mosaic

        labels9 = []
        s = self.img_size
        indices = [index] + random.choices(range(len(self.img_files)), k=8) # 8 additional image indices

        padded_size = (s * 3, s * 3)

        for i, index in enumerate(indices):
            img, (h_, w_), (h, w) = self.load_image(index)

            # place img in img9
            if i == 0:  # center
                img9 = np.zeros((s * 3, s * 3, img.shape[2]), dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

            # Labels
            pad = (pady, padx)
            targets = self.load_target(index, pad, (h_, w_), (h, w), padded_size)
            labels9.append(targets)

        # Concat labels
        labels9 = torch.cat(labels9, 0)

        # Offset
        yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]
        
        # Translate labels
        labels9[:, 2] -= xc
        labels9[:, 3] -= yc

        # Remove objects that exceed the size of images (the cropped area) when doing mosaic augmentation
        labels9 = self.filtering(labels9, img9.shape[:2])

        return img9, labels9

    def filtering(self, targets, img_size):
        # Remove objects that exceed the size of images (the cropped area) when doing mosaic augmentation
        height, width = img_size
        mask = (
            (targets[:, 2] > 0) & (targets[:, 2] < width) &
            (targets[:, 3] > 0) & (targets[:, 3] < height)
        )

        return targets[mask]
    
    def normalize(self, targets, img_size):
        # Normalize x, y, w, h, of targets into [0, 1]
        height, width = img_size
        targets[:, [2, 4]] /= width
        targets[:, [3, 5]] /= height

        return targets
    
    def load_files(self):
        raise NotImplementedError
