import torch
import time
import argparse
from tools.plot import load_class_names, plot_boxes
from tools.post_process import post_process
from tools.load import ImageDataset
from model.model import Yolo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/detect", help="path to dataset")
    parser.add_argument("--output_folder", type=str, default="outputs", help="path to outputs")
    parser.add_argument("--weights_path", type=str, default="weights/AOD_800.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.7, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    class_names = load_class_names(args.class_path)
    pretrained_dict = torch.load(args.weights_path, map_location=torch.device('cpu'))
    model = Yolo(n_classes=2)
    model = model.to(device)
    model.load_state_dict(pretrained_dict)

    model.eval()

    dataset = ImageDataset(args.image_folder, img_size=args.img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    boxes = []
    imgs = []

    start = time.time()
    for img_path, img in dataloader:
        img = torch.autograd.Variable(img.type(FloatTensor))

        with torch.no_grad():
            temp = time.time()
            output, _ = model(img)  # batch=1 -> [1, n, n], batch=3 -> [3, n, n]
            temp1 = time.time()
            box = post_process(output, args.conf_thres, args.nms_thres)
            temp2 = time.time()
            boxes.extend(box)
            print('-----------------------------------')
            num = 0
            for b in box:
                num += len(b)
            print("{}-> {} objects found".format(img_path, num))
            print("Inference time : ", round(temp1 - temp, 5))
            print("Post-processing time : ", round(temp2 - temp1, 5))
            print('-----------------------------------')

        imgs.extend(img_path)

    for i, (img_path, box) in enumerate(zip(imgs, boxes)):
        plot_boxes(img_path, box, class_names, args.img_size, args.output_folder)

    end = time.time()

    print('-----------------------------------')
    print("Total detecting time : ", round(end - start, 5))
    print('-----------------------------------')
