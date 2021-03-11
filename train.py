import time
import random
import argparse
import numpy as np
import torch
from terminaltables import AsciiTable
from torch.autograd import Variable

from model.model import Yolo
from tools.load import split_data
from tools.scheduler import CosineAnnealingWarmupRestarts
from tools.logger import *


def weights_init_normal(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def init():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", type=str, default="data/train", help="path to dataset")
    parser.add_argument("--weights_path", type=str, default="weights/yolov4_kun.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="size of batches")
    parser.add_argument("--subdivisions", type=int, default=4, help="size of mini batches")
    parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
    args = parser.parse_args()
    print(args)

    logger = Logger("logs")
    init()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained_dict = torch.load(args.weights_path)
    model = Yolo(n_classes=2)
    model = model.to(device)
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    # 第552項開始為yololayer，訓練時不需要用到
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    pretrained_dict = {k: v for i, (k, v) in enumerate(pretrained_dict.items()) if i < 552}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.apply(weights_init_normal)  # 權重初始化
    model.load_state_dict(model_dict)

    train_dataset, train_dataloader = split_data(args.train_folder, args.img_size, args.batch_size)
    num_iters_per_epoch = len(train_dataloader)
    scheduler_iters = round(args.epochs * len(train_dataloader) / args.subdivisions)
    total_step = num_iters_per_epoch * args.epochs

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=scheduler_iters,
                                              cycle_mult=1.0,
                                              max_lr=args.lr,
                                              min_lr=0,
                                              warmup_steps=round(scheduler_iters * 0.1),
                                              gamma=1.0)

    for epoch in range(args.epochs):
        total_loss = 0.0
        start_time = time.time()
        print("\n---- [Epoch %d/%d] ----\n" % (epoch + 1, args.epochs))
        model.train()

        for batch, (_, imgs, targets) in enumerate(train_dataloader):
            global_step = num_iters_per_epoch * epoch + batch + 1
            imgs = Variable(imgs.to(device), requires_grad=True)
            targets = Variable(targets.to(device), requires_grad=False)

            outputs, loss = model(imgs, targets)

            loss.backward()
            total_loss += loss.item()

            if global_step % args.subdivisions == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # ---------------------
            # -      logging      -
            # ---------------------
            tensorboard_log = []
            loss_table_name = ["Step: %d/%d" % (global_step, total_step),
                               "loss", "reg_loss", "conf_loss", "cls_loss"]
            loss_table = [loss_table_name]

            temp = ["YoloLayer1"]
            for name, metric in model.yolo1.metrics.items():
                if name in loss_table_name:
                    temp.append(metric)
                tensorboard_log += [(f"{name}_1", metric)]
            loss_table.append(temp)

            temp = ["YoloLayer2"]
            for name, metric in model.yolo2.metrics.items():
                if name in loss_table_name:
                    temp.append(metric)
                tensorboard_log += [(f"{name}_2", metric)]
            loss_table.append(temp)

            temp = ["YoloLayer3"]
            for name, metric in model.yolo3.metrics.items():
                if name in loss_table_name:
                    temp.append(metric)
                tensorboard_log += [(f"{name}_3", metric)]
            loss_table.append(temp)

            print(AsciiTable(loss_table).table)
            logger.list_of_scalars_summary(tensorboard_log, global_step)

        print("Total Loss: %f, Runtime %f" % (total_loss, time.time() - start_time))

    torch.save(model.state_dict(), "weights/yolov4_train.pth")
