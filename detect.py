import torch
import time
import os
import glob

from lib.options import DetectOptions
from lib.plot import plot_boxes
from lib.post_process import post_process
from lib.load import load_class_names
from datasets.base_dataset import ImageDataset
from model.yolo import Yolo

class Detect:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = load_class_names(os.path.join(self.args.data_folder, "class.names"))
        self.model = None

    def load_model(self):
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
        self.model = Yolo(n_classes=self.args.number_of_classes)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(pretrained_dict)

    def save_results(self, imgs, boxes):
        save_folder = os.path.join("outputs", self.args.model_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for i, (img_path, box) in enumerate(zip(imgs, boxes)):
            plot_boxes(img_path, box, self.class_names, self.args.img_size, save_folder)

    def detect(self):
        dataset = ImageDataset(os.path.join(self.args.data_folder, "detect"), img_size=self.args.img_size, ext=self.args.ext)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)

        self.load_model()
        self.model.eval()

        start = time.time()
        for img_path, img in dataloader:
            boxes, imgs = [], []

            img = img.to(self.device)

            with torch.no_grad():
                temp = time.time()
                output, _ ,_ = self.model(img,target = None)  # batch=1 -> [1, n, n], batch=3 -> [3, n, n]
                temp1 = time.time()
                box = post_process(output, self.args.conf_thres, self.args.nms_thres)
                temp2 = time.time()
                boxes.extend(box)
                print('-----------------------------------')
                num = 0
                for b in box:
                    if b is None:
                        break
                    num += len(b)
                print("{}-> {} objects found".format(img_path, num))
                print("Inference time : ", round(temp1 - temp, 5))
                print("Post-processing time : ", round(temp2 - temp1, 5))
                print('-----------------------------------')

            imgs.extend(img_path)
            self.save_results(imgs, boxes)

        end = time.time()

        print('-----------------------------------')
        print("Total detecting time : ", round(end - start, 5))
        print('-----------------------------------')


if __name__ == "__main__":
    parser = DetectOptions()
    args = parser.parse()
    print(args)

    d = Detect(args)
    d.detect()
