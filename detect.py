import torch
import time
import os
import glob
import yaml
import argparse
import cv2 as cv
import numpy as np

from lib.options import DetectOptions
from lib.plot import plot_boxes
from lib.post_process import post_process
from datasets.base_dataset import ImageDataset
from model.yolo import Yolo
from lib.logger import logger


class Detect:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def load_model(self, n_classes):
        model_path = os.path.join("weights", self.args.model_name)
        if os.path.exists(model_path):
            weight_path = glob.glob(os.path.join(model_path, "*.pth"))
            if len(weight_path) == 0:
                assert False, "Model weight not found"
            elif len(weight_path) > 1:
                assert False, "Multiple weights are found. Please keep only one weight in your model directory"
            else:
                weight_path = weight_path[0]
        else:
            assert False, "Model is not exist"
        pretrained_dict = torch.load(weight_path, map_location=self.device)
        self.model = Yolo(n_classes=n_classes)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(pretrained_dict)

    def save_results(self, imgs, boxes, class_names):
        save_folder = os.path.join("outputs", self.args.model_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for i, (img_path, box) in enumerate(zip(imgs, boxes)):
            plot_boxes(img_path, box, class_names, self.args.img_size, save_folder)

    def detect(self):
        # load data info
        with open(self.args.data, "r") as stream:
            data = yaml.safe_load(stream)

        dataset = ImageDataset(data["test"], img_size=self.args.img_size, ext=self.args.ext)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)

        self.load_model(len(data["names"]))
        self.model.eval()

        start = time.time()
        for img_path, img in dataloader:
            img = img.to(self.device)

            with torch.no_grad():
                temp = time.time()
                output, _  = self.model(img)  # batch=1 -> [1, n, n], batch=3 -> [3, n, n]
                temp1 = time.time()
                boxes = post_process(output,self.args.img_size, self.args.conf_thres, self.args.nms_thres)
                temp2 = time.time()

                logger.info('-----------------------------------')
                num = 0
                for b in boxes:
                    if b is None:
                        break
                    num += len(b)
                
                logger.info("{}-> {} objects found".format(img_path, num))
                logger.info(("Inference time : ") + ('%10.4g') % (round(temp1 - temp, 5)))
                logger.info(("Post-processing time : ") + ('%10.4g') % (round(temp2 - temp1, 5)))
                logger.info('-----------------------------------')

            self.save_results(img_path, boxes, data["names"])

        end = time.time()

        logger.info('-----------------------------------')
        logger.info(("Total detecting time : ") + ('%10.4g') % (round(end - start, 5)))
        logger.info('-----------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="", help=".yaml path for data")
    parser.add_argument("--model_name", type=str, default="UCAS_AOD", help="model name")
    parser.add_argument("--conf_thres", type=float, default=0.7, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
    parser.add_argument("--ext", type=str, default="png", choices=["png", "jpg"], help="Image file format")
    args = parser.parse_args()
    print(args)

    d = Detect(args)
    d.detect()
