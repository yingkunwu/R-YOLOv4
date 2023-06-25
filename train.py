import time
import random
import numpy as np
import torch
import os
import shutil
import json
import logging
from colorlog import ColoredFormatter

import tqdm

from model.yolo import Yolo
from lib.load import load_data
from lib.scheduler import CosineAnnealingWarmupRestarts
from lib.logger import *
from lib.options import TrainOptions
from lib.utils import load_class_names
from test import test


def setup_logger(log_file_path: str = None):
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter(
        "%(asctime)s %(log_color)s%(levelname)-8s %(filename)s[line:%(lineno)d]: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'blue',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        })

    logger = logging.getLogger(__name__)
    shandler = logging.StreamHandler()
    shandler.setFormatter(formatter)
    shandler.setLevel(level=logging.INFO)

    logger.addHandler(shandler)
    logger.setLevel(level=logging.INFO)
    return logger

logger = setup_logger()

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
        self.class_names = load_class_names(os.path.join(self.args.data_folder, "class.names"))
        self.model = None
        self.logger = None

    def check_model_path(self):
        if os.path.exists(self.model_path):
            while True:
                inp = input(f">> Model name exists, do you want to override the previous model? [Y:N]")
                if inp.lower()[0] == "y":
                    shutil.rmtree(self.model_path)
                    break
                elif inp.lower()[0] == "n":
                    print(">> Stop training!")
                    exit(0)
                    
        os.makedirs(self.model_path)
        os.makedirs(os.path.join(self.model_path, "logs"))

    def load_model(self):
        pretrained_dict = torch.load(self.args.weights_path)
        self.model = Yolo(n_classes=self.args.number_of_classes)
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
    
    def logging_processes(self, total_loss, epoch, global_step, total_step, start_time)->dict:
        tensorboard_log = {}


        loss_dict = {"cls_loss" : 0.0, "conf_loss": 0.0, "reg_loss": 0.0}


        for name, metric in self.model.yolo1.metrics.items():
            if name in loss_dict:
                loss_dict[name] += metric
            tensorboard_log[f"{name}_1"] = metric

        for name, metric in self.model.yolo2.metrics.items():
            if name in loss_dict:
                loss_dict[name] += metric
            tensorboard_log[f"{name}_2"] = metric

        for name, metric in self.model.yolo3.metrics.items():
            if name in loss_dict:
                loss_dict[name] += metric
            tensorboard_log[f"{name}_3"] = metric


        tensorboard_log["total_loss"] = total_loss
        self.logger.list_of_scalars_summary(tensorboard_log, global_step)

        return loss_dict

    def train(self):
        init()
        self.check_model_path()
        self.load_model()
        self.save_opts()
        self.logger = Logger(os.path.join(self.model_path, "logs"))

        augment = False if self.args.no_augmentation else True
        mosaic = False if self.args.no_mosaic else True
        multiscale = False if self.args.no_multiscale else True

        train_dataset, train_dataloader = load_data(self.args.data_folder, self.args.dataset, "train", self.args.img_size,
                                                    self.args.batch_size, augment=augment, mosaic=mosaic, multiscale=multiscale)
        num_iters_per_epoch = len(train_dataloader)

        scheduler_iters = round(self.args.epochs * len(train_dataloader) / self.args.subdivisions)
        total_step = num_iters_per_epoch * self.args.epochs

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                first_cycle_steps=round(scheduler_iters),
                                                max_lr=self.args.lr,
                                                min_lr=1e-5,
                                                warmup_steps=round(scheduler_iters * 0.1),
                                                cycle_mult=1,
                                                gamma=1)
        logger.info(f'Image sizes {self.args.img_size}')
        logger.info(f'Starting training for {self.args.epochs} epochs...')

        start_time = time.time()
        best_fitness = 0

        for epoch in range(self.args.epochs):
            # -------------------
            # ------ Train ------
            # -------------------
            self.model.train()
      
            logger.critical(('\n' + '%10s' * 6) % ('Epoch', 'box_loss', 'obj_loss', 'cls_loss', 'total', 'img_size'))
            pbar = enumerate(train_dataloader)
            pbar = tqdm.tqdm(pbar, total=len(train_dataloader))
            #pbar = enumerate(tqdm.tqdm(train_dataloader))
            for batch, (_, imgs, targets) in pbar:
                global_step = num_iters_per_epoch * epoch + batch + 1
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                outputs, loss, loss_items = self.model(imgs, targets)

                loss.backward()
                total_loss = loss.detach().item()

                if global_step % self.args.subdivisions == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                loss_dict = self.logging_processes(total_loss, epoch, global_step, total_step, start_time)
                
                s = ('%10s'  + '%10.4g' * 5) % (
                    '%g/%g' % (epoch, self.args.epochs),  loss_dict["reg_loss"],
                    loss_dict["conf_loss"],loss_dict["cls_loss"], total_loss, imgs.shape[-1])

                pbar.set_description(s)
                pbar.update(0)

            # -------------------
            # ------ Valid ------
            # -------------------
            mp, mr, map50, map, loss, loss_items = test(self.model, self.device, self.class_names, self.args.data_folder, 
                                self.args.dataset, self.args.img_size, self.args.batch_size * 2, conf_thres=0.001, nms_thres=0.65)

            fit = fitness(np.array([mp, mr, map50, map]))
            if fit > best_fitness:
                best_fitness = fit
                self.save_model("best")
                logger.info("Current best model is saved!")
            self.save_model("last")

        logger.info("Done!")

        
if __name__ == "__main__":
    parser = TrainOptions()
    args = parser.parse()
    print(args)

    t = Train(args)
    t.train()
