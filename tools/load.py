# Reference: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/utils/datasets.py

import glob
import random
import os
import numpy as np
import cv2 as cv
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from tools.plot import xywha2xyxyxyxy
from tools.augments import vertical_flip, horisontal_flip, rotate, hsv, gaussian_noise, mixup


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
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.png" % folder_path))
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


class ListDataset(Dataset):
    def __init__(self, list_path, labels, img_size=416, augment=True, mosaic=True, multiscale=True, normalized_labels=False, custom=False):
        self.img_files = list_path

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.labels = labels
        self.augment = augment
        self.mosaic = mosaic
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.custom = custom
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
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
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

    def load_target(self, index, h_factor, w_factor, pad, padded_h, padded_w, mosaic=False):
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        if os.path.exists(label_path):
            if self.custom:
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

            else:
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


def split_data(data_dir, img_size, batch_size=4, shuffle=True, augment=True, mosaic=True, multiscale=True, custom=False):
    if custom:
        inputs = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        labels = None
    
    else:
        dataset = ImageFolder(data_dir)

        classes = [[] for _ in range(len(dataset.classes))]

        for x, y in dataset.samples:
            classes[int(y)].append(x)

        inputs, labels = [], []

        for i, data in enumerate(classes):  # 讀取每個類別中所有的檔名 (i: label, data: filename)

            for x in data:
                inputs.append(x)
                labels.append(i)

    dataset = ListDataset(inputs, labels, img_size=img_size, augment=augment, mosaic=mosaic, multiscale=multiscale, custom=custom)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                   pin_memory=True, collate_fn=dataset.collate_fn)

    return dataset, dataloader


if __name__ == "__main__":
    train_dataset, train_dataloader = split_data("data/test", 608, batch_size=1, mosaic=True, multiscale=False, custom=False)

    for i, (img_path, imgs, targets) in enumerate(train_dataloader):
        img = imgs.squeeze(0).numpy().transpose(1, 2, 0)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        for p in targets:
            x, y, w, h, theta = p[2] * img.shape[1], p[3] * img.shape[1], p[4] * img.shape[1], p[5] * img.shape[1], p[6]

            X1, Y1, X2, Y2, X3, Y3, X4, Y4 = xywha2xyxyxyxy(torch.tensor([x, y, w, h, theta]))
            X1, Y1, X2, Y2, X3, Y3, X4, Y4 = int(X1), int(Y1), int(X2), int(Y2), int(X3), int(Y3), int(X4), int(Y4)

            cv.line(img, (X1, Y1), (X2, Y2), (255, 0, 0), 1)
            cv.line(img, (X2, Y2), (X3, Y3), (255, 0, 0), 1)
            cv.line(img, (X3, Y3), (X4, Y4), (255, 0, 0), 1)
            cv.line(img, (X4, Y4), (X1, Y1), (255, 0, 0), 1)

        img[:, 1:] = img[:, 1:] * 255.0
        if img_path[0].split('/')[-2] == str(1):
            path = "data/plane_" + img_path[0].split('/')[-1]
        else:
            path = "data/car_" + img_path[0].split('/')[-1]
        cv.imwrite(path, img)
        #cv.imshow('My Image', img)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
