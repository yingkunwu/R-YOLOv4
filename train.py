import time
import random
import numpy as np
import torch
import os
import shutil
import json
import tqdm
import yaml
import argparse

from model.yolo import Yolo
from lib.load import load_data
from lib.scheduler import CosineAnnealingWarmupRestarts
from lib.logger import Logger, logger
from lib.options import TrainOptions
from lib.loss import ComputeLoss
from test import test
from torch.optim.lr_scheduler import CosineAnnealingLR


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


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x * w).sum(0)


class Train:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = os.path.join("weights", self.args.model_name)
        self.model = None
        self.logger = None

    def check_model_path(self):
        if os.path.exists(self.model_path):
            while True:
                logger.warning("Model name exists, do you want to override the previous model?")
                inp = input(">> [Y:N]")
                if inp.lower()[0] == "y":
                    shutil.rmtree(self.model_path)
                    break
                elif inp.lower()[0] == "n":
                    logger.info("Stop training!")
                    exit(0)
                    
        os.makedirs(self.model_path)
        os.makedirs(os.path.join(self.model_path, "logs"))

    def load_model(self):
        pretrained_dict = torch.load(self.args.weights_path)
        self.model = Yolo(n_classes=2)
        self.model = self.model.to(self.device)
        model_dict = self.model.state_dict()

        # 1. filter out unnecessary keys
        # 第552項開始為yololayer，訓練時不需要用到
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        pretrained_dict = {k: v for i, (k, v) in enumerate(pretrained_dict.items()) if i < 552}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.model.apply(weights_init_normal)  # 權重初始化
        self.model.load_state_dict(model_dict)

    def save_model(self, postfix):
        save_folder = os.path.join(self.model_path, "ryolov4_{}.pth".format(postfix))
        torch.save(self.model.state_dict(), save_folder)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        to_save = self.args.__dict__.copy()
        with open(os.path.join(self.model_path, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)
    
    def logging_processes(self, loss_items, epoch,mean_recall = None,mean_precision = None ,map50 = None, map5095 = None, val_mode = False)->None:
        tensorboard_log = {}
        # update in validation 
        if val_mode and map50 != None and map5095 != None:
            for name, metric in loss_items.items():
                tensorboard_log[f"val/{name}"] = metric

            tensorboard_log["mean recall"] = mean_recall
            tensorboard_log["mean precision"] = mean_precision
            tensorboard_log["mAP@.5"] = map50
            tensorboard_log["mAP@.5:.95"] = map5095
        # update in training 
        else:
            for name, metric in loss_items.items():
                tensorboard_log[f"train/{name}"] = metric

        self.logger.list_of_scalars_summary(tensorboard_log, epoch)

    def train(self):
        self.check_model_path()
        self.load_model()
        self.save_opts()
        self.logger = Logger(os.path.join(self.model_path, "logs"))

        # load data info
        with open(self.args.data, "r") as stream:
            data = yaml.safe_load(stream)

        # load hyperparameters
        with open(self.args.hyp, "r") as stream:
            hyp = yaml.safe_load(stream)

        train_dataset, train_dataloader = load_data(
            data['train'], data['names'], data['type'], hyp, self.args.img_size, self.args.batch_size, augment=True
        )
        num_iters_per_epoch = len(train_dataloader)

        subdivisions = 64

        scheduler_iters = round(self.args.epochs * len(train_dataloader) / subdivisions)
        total_step = num_iters_per_epoch * self.args.epochs

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        # scheduler = CosineAnnealingWarmupRestarts(optimizer,
        #                                         first_cycle_steps=round(scheduler_iters),
        #                                         max_lr=self.args.lr,
        #                                         min_lr=1e-5,
        #                                         warmup_steps=round(scheduler_iters * 0.1),
        #                                         cycle_mult=1,
        #                                         gamma=1)
        scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_iters, eta_min=1e-5)
        
        compute_loss = ComputeLoss(hyp)

        logger.info(f'Image sizes {self.args.img_size}')
        logger.info(f'Starting training for {self.args.epochs} epochs...')
        

        start_time = time.time()
        best_fitness = 0

        for epoch in range(self.args.epochs):
            # -------------------
            # ------ Train ------
            # -------------------
            self.model.train()
      
            logger.info(('\n' + '%10s' * 7) % ('Epoch', 'last lr', 'box_loss', 'obj_loss', 'cls_loss', 'total', 'img_size'))
            pbar = enumerate(train_dataloader)
            pbar = tqdm.tqdm(pbar, total=len(train_dataloader))
            #pbar = enumerate(tqdm.tqdm(train_dataloader))
            for batch, (_, imgs, targets) in pbar:
                global_step = num_iters_per_epoch * epoch + batch + 1
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                #outputs, loss, loss_items = self.model(imgs, targets)
                outputs, masked_anchors = self.model(imgs)
                loss, loss_items = compute_loss(outputs, targets, masked_anchors)

                loss.backward()
                total_loss = loss.detach().item()

                # TODO: make sure weight updating process
                if global_step % subdivisions == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                
                s = ('%10s'  + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, self.args.epochs),scheduler.get_last_lr()[0],  loss_items["reg_loss"],
                    loss_items["conf_loss"], loss_items["cls_loss"], total_loss, imgs.shape[-1])

                pbar.set_description(s)
                pbar.update(0)
            # update the training log for tensorboard every epoch    
            self.logging_processes(loss_items, epoch)

            # -------------------
            # ------ Valid ------
            # -------------------
            mp, mr, map50, map, loss, loss_items = test(
                self.model, compute_loss, self.device, data, hyp, 
                self.args.img_size, self.args.batch_size * 2, conf_thres=0.001, nms_thres=0.65
            )

            self.logging_processes(loss_items, epoch, mean_recall = mr, mean_precision = mp ,map50 = map50, map5095 = map, val_mode = True)

            fit = fitness(np.array([mp, mr, map50, map]))
            if fit > best_fitness:
                best_fitness = fit
                self.save_model("best")
                logger.info("Current best model is saved!")
            self.save_model("last")

        logger.info("Done!")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="size of batches")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--weights_path", type=str, default="weights/pretrained/yolov4.pth", help="path to pretrained weights file")
    parser.add_argument("--model_name", type=str, default="trash", help="new model name")
    parser.add_argument("--data", type=str, default="", help=".yaml path for data")
    parser.add_argument("--hyp", type=str, default="", help=".yaml path for hyperparameters")

    args = parser.parse_args()
    print(args)

    init()
    t = Train(args)
    t.train()
