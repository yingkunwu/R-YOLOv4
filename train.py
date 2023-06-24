import time
import random
import numpy as np
import torch
import os
import shutil
import json
from terminaltables import AsciiTable
import logging
import tqdm

from model.yolo import Yolo
from lib.load import load_data
from lib.scheduler import CosineAnnealingWarmupRestarts
from lib.logger import *
from lib.options import TrainOptions

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(message)s", level=logging.INFO)


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

        # TODO: 以這個為格式，希望可以把每個epoch的資訊都顯示在這些title下面，然後一直到下一個epoch才換行，這邊其實也想參考yolov7的寫法，可以的話你可以先跑一次yolov7的看看，或是我跑然後螢幕錄影給你看。我已經有從153行開始改了，你可以從那邊繼續。
        #logger.info(('\n' + '%10s' * 6) % ('Epoch', 'box_loss', 'obj_loss', 'cls_loss', 'total', 'img_size'))

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
        #print(log)

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

        start_time = time.time()

        self.model.train()
        for epoch in range(self.args.epochs):
            # -------------------
            # ------ Train ------
            # -------------------
            pbar = tqdm.tqdm(enumerate(train_dataloader))
            for batch, (_, imgs, targets) in pbar:
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
                #s = ('%10s' * 2 + '%10.4g' * 6) % (
                #    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                #pbar.set_description("I am dick")
        
            self.save_model()
            print("Model is saved!")

            # -------------------
            # ------ Valid ------
            # -------------------
            """self.model.eval()
            pbar = tqdm.tqdm(enumerate(train_dataloader))
            for batch, (_, imgs, targets) in pbar:
                with torch.no_grad():
                    global_step = num_iters_per_epoch * epoch + batch + 1
                    imgs = imgs.to(self.device)
                    targets = targets.to(self.device)

                    outputs, loss = self.model(imgs, targets)
                    outputs = post_process(outputs, conf_thres=self.args.conf_thres, nms_thres=self.args.nms_thres)

                    loss.backward()
                    total_loss = loss.detach().item()

                    if global_step % self.args.subdivisions == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
        
            self.save_model()
            print("Model is saved!")"""

        print("Done!")

if __name__ == "__main__":
    parser = TrainOptions()
    args = parser.parse()
    print(args)

    t = Train(args)
    t.train()
