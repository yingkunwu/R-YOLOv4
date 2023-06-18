import torch
import numpy as np
import random
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms

def gaussian_noise(image, mean=0, var=100.0):
    var = random.uniform(0, var)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    image = image.astype("float32")
    out = image + noise
    out = np.clip(out, 0.0, 255.0)
    return out.astype(np.uint8)


def hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=img)


def mixup(img, labels, img2, labels2):
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    img = (img * r + img2 * (1 - r)).astype(np.uint8)
    labels = torch.cat((labels, labels2), 0)
    return img, labels


def vertical_flip(images, targets):
    images = torch.flip(images, [1])
    targets[:, 3] = 1 - targets[:, 3]
    targets[:, 6] = - targets[:, 6]

    return images, targets


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    targets[:, 6] = - targets[:, 6]

    return images, targets


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


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

    images = images.unsqueeze(0)
    rot_mat = get_rot_mat(radian)[None, ...].repeat(images.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, images.size(), align_corners=True)
    images = F.grid_sample(images, grid, align_corners=True)
    images = images.squeeze(0)
    # x,y of targets
    points = torch.cat([targets[:, 2:4], torch.ones(len(targets), 1)], dim=1)

    points = points.T
    points = torch.matmul(T2, torch.matmul(R, torch.matmul(T1, points))).T
    targets[:, 2:4] = points[:, :2]

    # throwing away the bbox label of those surpass the boundary
    targets = targets[targets[:, 2] < 1]
    targets = targets[targets[:, 2] > 0]
    targets = targets[targets[:, 3] < 1]
    targets = targets[targets[:, 3] > 0]
    assert (targets[:, 2:4] > 0).all() or (targets[:, 2:4] < 1).all()

    targets[:, 6] = targets[:, 6] - radian
    targets[:, 6][targets[:, 6] <= -np.pi / 2] = targets[:, 6][targets[:, 6] <= -np.pi / 2] + np.pi

    assert (-np.pi / 2 < targets[:, 6]).all() or (targets[:, 6] <= np.pi / 2).all()

    return images, targets


def random_warping(images, targets, scale = .5, translate = .1):
    c, h, w = images.shape[0], images.shape[1], images.shape[2]

    images = images.numpy()
    images = np.swapaxes(images,0,1)
    images = np.swapaxes(images,1,2)

    # Rotation(Scaling)

    R = np.eye(3)
    # a = random.uniform(-180, 90)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations

    s = random.uniform(1 - scale, 1 + scale + 0.2)

    R[:2] = cv2.getRotationMatrix2D(angle=0, center=(w//2, h//2), scale=s)

    # Translation

    T = np.eye(3)
    T[0, 2] = random.uniform(0.3 - translate, 0.3 + translate) * w  # x translation (pixels)
    T[1, 2] = random.uniform(0.3 - translate, 0.3 + translate) * h  # y translation (pixels)

    M = T @ R
    output = cv2.warpPerspective(images, M, dsize=(w, h), borderValue=(0, 0, 0))

    output = np.swapaxes(output,1,2)
    output = np.swapaxes(output,0,1)
    output = torch.tensor(output)

    M = torch.tensor(M,dtype=torch.double)
    M[0, 2] = M[0, 2] / w
    M[1, 2] = M[1, 2] / h

    xy = targets[:, 2:4] # num of target x 2
    xy = torch.cat((xy, torch.ones(xy.size()[0]).view(xy.size()[0],1)),dim = -1).double() # num of target x 3 
    
    new_xy = (torch.matmul(M, xy.t())).t()[:, :2]

    wh = targets[:, 4:6]
    new_wh = wh * s

    targets[:, 2:4] = new_xy
    targets[:, 4:6] = new_wh

    targets = targets[targets[:, 2] < 1]
    targets = targets[targets[:, 2] > 0]
    targets = targets[targets[:, 3] < 1]
    targets = targets[targets[:, 3] > 0]

    return output, targets
    #pass