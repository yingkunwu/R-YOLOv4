import time
import random
import numpy as np
import torch
import os
import shutil
import json
from terminaltables import AsciiTable

from model.model import Yolo
from tools.load import split_data
from tools.scheduler import CosineAnnealingWarmupRestarts
from tools.logger import *
from tools.options import TrainOptions


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


class Train:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = os.path.join("weights", self.args.model_name)
        self.model = None
        self.logger = None

    def check_model_path(self):
        if os.path.exists(self.model_path):
            inp = input(f">> Model name exists, do you want to override model name ? [y:N]")
            if inp.lower()[0] == "y":
                shutil.rmtree(self.model_path)
            else:
                print(">> Stop training!")
                exit(1)
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

    def save_model(self):
        save_folder = os.path.join(self.model_path, "ryolov4.pth")
        torch.save(self.model.state_dict(), save_folder)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        to_save = self.args.__dict__.copy()
        with open(os.path.join(self.model_path, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def log(self, total_loss, epoch, global_step, total_step, start_time):
        log = "\n---- [Epoch %d/%d] ----\n" % (epoch + 1, self.args.epochs)

        tensorboard_log = {}
        loss_table_name = ["Step: %d/%d" % (global_step, total_step),
                            "loss", "reg_loss", "conf_loss", "cls_loss"]
        loss_table = [loss_table_name]

        temp = ["YoloLayer1"]
        for name, metric in self.model.yolo1.metrics.items():
            if name in loss_table_name:
                temp.append(metric)
            tensorboard_log[f"{name}_1"] = metric
        loss_table.append(temp)

        temp = ["YoloLayer2"]
        for name, metric in self.model.yolo2.metrics.items():
            if name in loss_table_name:
                temp.append(metric)
            tensorboard_log[f"{name}_2"] = metric
        loss_table.append(temp)

        temp = ["YoloLayer3"]
        for name, metric in self.model.yolo3.metrics.items():
            if name in loss_table_name:
                temp.append(metric)
            tensorboard_log[f"{name}_3"] = metric
        loss_table.append(temp)

        tensorboard_log["total_loss"] = total_loss
        self.logger.list_of_scalars_summary(tensorboard_log, global_step)

        log += AsciiTable(loss_table).table
        log += "\nTotal Loss: %f, Runtime: %f\n" % (total_loss, time.time() - start_time)
        print(log)

    def train(self):
        init()
        self.check_model_path()
        self.load_model()
        self.save_opts()
        self.logger = Logger(os.path.join(self.model_path, "logs"))

        augment = False if self.args.no_augmentation else True
        multiscale = False if self.args.no_multiscale else True

        train_dataset, train_dataloader = split_data(self.args.train_path, self.args.img_size, self.args.batch_size, 
                                                        augment=augment, multiscale=multiscale)
        num_iters_per_epoch = len(train_dataloader)
        scheduler_iters = round(self.args.epochs * len(train_dataloader) / self.args.subdivisions)
        total_step = num_iters_per_epoch * self.args.epochs

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                first_cycle_steps=scheduler_iters,
                                                cycle_mult=1.0,
                                                max_lr=self.args.lr,
                                                min_lr=0,
                                                warmup_steps=round(scheduler_iters * 0.1),
                                                gamma=1.0)

        start_time = time.time()
        self.model.train()
        for epoch in range(self.args.epochs):

            for batch, (_, imgs, targets) in enumerate(train_dataloader):

                global_step = num_iters_per_epoch * epoch + batch + 1
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                outputs, loss = self.model(imgs, targets)

                loss.backward()
                total_loss = loss.detach().item()

                if global_step % self.args.subdivisions == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                self.log(total_loss, epoch, global_step, total_step, start_time)
        
            self.save_model()
            print("Model is saved!")

        print("Done!")

if __name__ == "__main__":
    parser = TrainOptions()
    args = parser.parse()
    print(args)

    t = Train(args)
    t.train()
