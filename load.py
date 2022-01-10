import glob
import os
import cv2 as cv
import torch
from torchvision.datasets import ImageFolder

from datasets.custom_dataset import CustomDataset
from datasets.UCASAOD_dataset import UCASAODDataset


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def split_data(data_dir, img_size, batch_size=4, shuffle=True, augment=True, mosaic=True, multiscale=True, custom=False):
    if custom:
        img_files = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        dataset = CustomDataset(img_files, None, img_size=img_size, augment=augment, mosaic=mosaic, multiscale=multiscale)
    else:
        dataset = ImageFolder(data_dir)

        classes = [[] for _ in range(len(dataset.classes))]

        for x, y in dataset.samples:
            classes[int(y)].append(x)

        img_files, labels = [], []

        for i, data in enumerate(classes):  # 讀取每個類別中所有的檔名 (i: label, data: filename)

            for x in data:
                img_files.append(x)
                labels.append(i)
        dataset = UCASAODDataset(img_files, labels, img_size=img_size, augment=augment, mosaic=mosaic, multiscale=multiscale)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                   pin_memory=True, collate_fn=dataset.collate_fn)

    return dataset, dataloader


if __name__ == "__main__":
    from utils.plot import xywha2xyxyxyxy
    #train_dataset, train_dataloader = split_data("data/trash_train", 608, batch_size=1, mosaic=True, multiscale=False, custom=True)
    train_dataset, train_dataloader = split_data("data/test_200", 608, batch_size=1, mosaic=True, multiscale=False, custom=False)

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

        #img[:, 1:] = img[:, 1:] * 255.0
        if img_path[0].split('/')[-2] == str(1):
            path = "data/plane_" + img_path[0].split('/')[-1]
        else:
            path = "data/car_" + img_path[0].split('/')[-1]
        #cv.imwrite(path, img)
        cv.imshow('My Image', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
