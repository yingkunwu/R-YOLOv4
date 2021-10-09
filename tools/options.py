import argparse

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--data_folder", type=str, default="data/train", help="path to train dataset")
        self.parser.add_argument("--weights_path", type=str, default="weights/pretrained/yolov4.pth", help="path to pretrained weights file")
        self.parser.add_argument("--model_name", type=str, default="new_model", help="new model name")
        self.parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
        self.parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
        self.parser.add_argument("--batch_size", type=int, default=2, help="size of batches")
        self.parser.add_argument("--subdivisions", type=int, default=8, help="size of mini batches")
        self.parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
        self.parser.add_argument("--number_of_classes", type=int, default=2, help="number of your output classes")
        self.parser.add_argument("--no_augmentation", action="store_true", help="if set, disable data augmentation in training")
        self.parser.add_argument("--no_multiscale", action="store_true", help="if set, disable multiscale data in training")

    def parse(self):
        return self.parser.parse_args()

class TestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--data_folder", type=str, default="data/test", help="path to dataset")
        self.parser.add_argument("--model_name", type=str, default="UCAS_AOD", help="model name")
        self.parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
        self.parser.add_argument("--conf_thres", type=float, default=0.7, help="object confidence threshold")
        self.parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
        self.parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold for evaluation")
        self.parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
        self.parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
        self.parser.add_argument("--number_of_classes", type=int, default=2, help="number of your output classes")


    def parse(self):
        return self.parser.parse_args()

class DetectOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--data_folder", type=str, default="data/detect", help="path to dataset")
        self.parser.add_argument("--model_name", type=str, default="UCAS_AOD", help="model name")
        self.parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
        self.parser.add_argument("--conf_thres", type=float, default=0.7, help="object confidence threshold")
        self.parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
        self.parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
        self.parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
        self.parser.add_argument("--number_of_classes", type=int, default=2, help="number of your output classes")

    def parse(self):
        return self.parser.parse_args()
