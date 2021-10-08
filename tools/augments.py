import torch
import numpy as np
import random
import cv2
import torch.nn.functional as F


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def gaussian_noise(images, mean, std):
    return images + (torch.randn(images.size()) * 0.1) * std + mean


def rand_uniform(min, max):
    if min > max:
        swap = min
        min = max
        max = swap
    return random.random() * (max - min) + min


def rand_scale(s):
    scale = rand_uniform(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale


def hsv(img):
    hue = rand_uniform(-0.1, 0.1)
    sat = rand_scale(1.3)
    exp = rand_scale(1.3)

    #print("hue: {}, sat: {}, exp: {}".format(hue, sat, exp))

    img = img.numpy().transpose((1, 2, 0))
    hsv_src = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)  # RGB to HSV
    hsv = cv2.split(hsv_src)
    hsv[1] *= sat
    hsv[2] *= exp
    hsv[0] += 60 * hue
    hsv_src = cv2.merge(hsv)
    img = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)  # HSV to RGB (the same as previous)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img


def rotate(images, targets):
    degree = np.random.rand() * 90
    radian = np.pi / 180 * degree
    R = torch.stack([
        torch.stack([torch.cos(torch.tensor(-radian)), -torch.sin(torch.tensor(-radian)), torch.tensor(0)]),
        torch.stack([torch.sin(torch.tensor(-radian)), torch.cos(torch.tensor(-radian)), torch.tensor(0)]),
        torch.stack([torch.tensor(0), torch.tensor(0), torch.tensor(1)])]).reshape(3, 3)
    T1 = torch.stack([
        torch.stack([torch.tensor(1), torch.tensor(0), torch.tensor(-0.5)]),
        torch.stack([torch.tensor(0), torch.tensor(1), torch.tensor(-0.5)]),
        torch.stack([torch.tensor(0), torch.tensor(0), torch.tensor(1)])]).reshape(3, 3)
    T2 = torch.stack([
        torch.stack([torch.tensor(1), torch.tensor(0), torch.tensor(0.5)]),
        torch.stack([torch.tensor(0), torch.tensor(1), torch.tensor(0.5)]),
        torch.stack([torch.tensor(0), torch.tensor(0), torch.tensor(1)])]).reshape(3, 3)
    #images = torch.rot90(images, 1, [1, 2])
    images = images.unsqueeze(0)
    rot_mat = get_rot_mat(radian)[None, ...].repeat(images.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, images.size(), align_corners=True)
    images = F.grid_sample(images, grid, align_corners=True)
    images = images.squeeze(0)

    points = torch.cat([targets[:, 2:4], torch.ones(len(targets), 1)], dim=1)
    points = points.T
    points = torch.matmul(T2, torch.matmul(R, torch.matmul(T1, points))).T
    targets[:, 2:4] = points[:, :2]

    targets = targets[targets[:, 2] < 1]
    targets = targets[targets[:, 2] > 0]
    targets = targets[targets[:, 3] < 1]
    targets = targets[targets[:, 3] > 0]
    assert (targets[:, 2:4] > 0).all() or (targets[:, 2:4] < 1).all()

    targets[:, 6] = targets[:, 6] - radian
    targets[:, 6][targets[:, 6] <= -np.pi / 2] = targets[:, 6][targets[:, 6] <= -np.pi / 2] + np.pi

    #for t in targets:
    #    if t[6] > 0:
    #        temp1, temp2 = t[4].clone(), t[5].clone()
    #        t[5], t[4] = temp1, temp2
    #        t[6] = t[6] - np.pi / 2
    #    elif t[6] <= -np.pi / 2:
    #        temp1, temp2 = t[4].clone(), t[5].clone()
    #        t[5], t[4] = temp1, temp2
    #        t[6] = t[6] + np.pi / 2

    assert (-np.pi / 2 < targets[:, 6]).all() or (targets[:, 6] <= np.pi / 2).all()
    return images, targets


def vertical_flip(images, targets):
    images = torch.flip(images, [0, 1])
    targets[:, 3] = 1 - targets[:, 3]
    targets[:, 6] = - targets[:, 6]

    return images, targets


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    targets[:, 6] = - targets[:, 6]

    return images, targets