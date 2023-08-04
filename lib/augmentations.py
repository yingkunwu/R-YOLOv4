import torch
import numpy as np
import random
import cv2
import torch.nn.functional as F


def hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)


def mixup(img, labels, img2, labels2):
    r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
    img = (img * r + img2 * (1 - r)).astype(np.uint8)
    labels = torch.cat((labels, labels2), 0)
    return img, labels


def vertical_flip(images, targets):
    images = np.flipud(images)
    targets[:, [3, 5, 7, 9]] = 1 - targets[:, [3, 5, 7, 9]]
    return images, targets


def horizontal_flip(images, targets):
    images = np.fliplr(images)
    targets[:, [2, 4, 6, 8]] = 1 - targets[:, [2, 4, 6, 8]]
    return images, targets


def random_warping(images, targets, degrees=10, scale=.9, translate=.1, border=(0, 0)):
    height = images.shape[0] + border[0] * 2  # shape(h, w, c)
    width = images.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -images.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -images.shape[0] / 2  # y translation (pixels)

    # Rotation and Scaling
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1.1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.3 - translate, 0.3 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.3 - translate, 0.3 + translate) * height  # y translation (pixels)

    M = T @ R @ C
    output = cv2.warpPerspective(images, M, dsize=(width, height), borderValue=(114, 114, 114))

    M = torch.tensor(M, dtype=torch.double)

    xyxyxyxy = targets[:, 2:] # x1, y1, x2, y2, x3, y3, x4, y4
    xyxyxyxy = xyxyxyxy.reshape(-1, 2)
    xyxyxyxy = torch.cat((xyxyxyxy, torch.ones(xyxyxyxy.size()[0]).view(xyxyxyxy.size()[0],1)), dim = -1).double()
    
    xyxyxyxy = (torch.matmul(M, xyxyxyxy.t())).t()[:, :2]
    targets[:, 2:] = xyxyxyxy.reshape(-1, 8)

    return output, targets
