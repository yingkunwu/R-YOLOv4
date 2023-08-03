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
    targets[:, 3] = 1 - targets[:, 3]
    targets[:, 6] = - targets[:, 6]
    return images, targets


def horizontal_flip(images, targets):
    images = np.fliplr(images)
    targets[:, 2] = 1 - targets[:, 2]
    targets[:, 6] = - targets[:, 6]
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
    output = cv2.warpPerspective(images, M, dsize=(width, height), borderValue=(0, 0, 0))

    M = torch.tensor(M, dtype=torch.double)

    xy = targets[:, 2:4] # num of target x 2
    xy = torch.cat((xy, torch.ones(xy.size()[0]).view(xy.size()[0],1)), dim = -1).double() # num of target x 3 
    
    new_xy = (torch.matmul(M, xy.t())).t()[:, :2]

    wh = targets[:, 4:6]
    new_wh = wh * s

    targets[:, 2:4] = new_xy
    targets[:, 4:6] = new_wh

    theta = targets[:, 6] + a * np.pi / 180
    for i in range(len(theta)):
        t = theta[i]
        if t > np.pi / 2:
            t -= np.pi
        elif t <= -np.pi / 2:
            t += np.pi
        theta[i] = t
    targets[:, 6] = theta

    # Check whether theta of oriented bounding boxes are within the defined range or not
    assert np.logical_and(-np.pi / 2 < theta, theta <= np.pi / 2).all(), \
        ("Theta of oriented bounding boxes are not within the boundary (-pi / 2, pi / 2]")

    return output, targets
