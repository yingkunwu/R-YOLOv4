import cv2 as cv
import numpy as np
import os
import yaml

from lib.load import load_data
from lib.plot import xywha2xyxyxyxy


if __name__ == "__main__":
    # load hyperparameters
    with open("data/hyp.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # load data info
    with open("data/UCAS_AOD.yaml", "r") as stream:
        data = yaml.safe_load(stream)
    
    train_dataset, train_dataloader = load_data(data['train'], data['names'], data['type'], config["hyp"], 608, 1, augment=False)

    for i, (img_path, imgs, targets) in enumerate(train_dataloader):
        print(targets)
        img = imgs.squeeze(0).numpy().transpose(1, 2, 0)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        targets = np.array(targets)

        for p in targets:
            x, y, w, h, theta = p[2] * img.shape[1], p[3] * img.shape[1], p[4] * img.shape[1], p[5] * img.shape[1], p[6]

            bbox = xywha2xyxyxyxy((x, y, w, h, theta))
            bbox = np.int0(bbox)
            cv.drawContours(img, [bbox], 0, (255, 0, 0), 1)

        cv.imshow('My Image', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

        img[:, 1:] = img[:, 1:] * 255.0
        output_path = os.path.join("outputs", "display")
        if not os.path.exists(output_path):
            os.mkdir(output_path) 
        filename = os.path.join(output_path, img_path[0].split('/')[-1])
        cv.imwrite(filename, img)
