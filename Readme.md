# R-YOLOv4

This is a PyTorch-based R-YOLOv4 implementation which combines YOLOv4 model and loss function from R3Det for arbitrary oriented object detection.
(Final project for NCKU INTRODUCTION TO ARTIFICIAL INTELLIGENCE course)

### Introduction
The objective of this project is to provide a capability of oriented object detection for YOLOv4 model. As a result, modifying the original loss function of bounding boxes for the model is needed. At last, I got a successful result by increasing the number of anchor boxes with different rotating angle and combining smooth-L1-IoU loss function proposed by [R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object](https://arxiv.org/abs/1908.05612) into the original loss.

### Dataset

**UCAS-High Resolution Aerial Object Detection Dataset (UCAS-AOD)**

Label: x1, y1, x2, y2, x3, y3, x4, y4, theta, x, y, width, height </br>
(x1, y1) is the coordinate located on the upper left of the bounding box; (x2, y2), (x3, y3) and (x4, y4) are the rest of the bounding box corners following the clockwise order.

Though it provides theta for each bounding box, it is not within the angle range that I want. You can check out how I calculated the angle that I need in [tools/load.py](https://github.com/kunnnnethan/R-YOLOv4/blob/bc106d5d4963a3085e8b88eb162b64d44e92abc3/tools/load.py).

### Features

---
#### Loss Function (only for x, y, w, h, theta)

<img src="https://i.imgur.com/zdA9RJj.png" alt="loss" height="90"/>
<img src="https://i.imgur.com/Qi1XFXS.png" alt="angle" height="70"/>

---
#### Scheduler
Cosine Annealing with Warmup (Reference: [Cosine Annealing with Warmup for PyTorch](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup))
</br>
<img src="https://i.imgur.com/qvTnszY.png" alt="scheduler" height="300"/>

---

#### Recall

<img src="https://i.imgur.com/mQf4S1m.png" alt="recall" height="300"/>

As the paper suggested, I get a better results from **f(ariou) = exp(1-ariou)-1**. Therefore I used it for my loss function.


### Usage

1. Clone and Setup Environment
```
$ git clone https://github.com/kunnnnethan/R-YOLOv4.git
$ cd R-YOLOv4/
$ python3.8 -m venv (your environment name)
$ source ~/your-environment-name/bin/activate
$ pip3 install torch torchvision torchaudio
$ pip install -r requirements.txt
```

2. Download  weights
```
$ ./setup/setup.sh
```
* Or Download it Manually

    [yolov4 pretrained weights](https://drive.google.com/uc?export=download&id=1sVD2d_y9VDirA-XOdcVDKCDrQw3e7ZJY)</br>
    [weight trained by UCAS_AOD dataset](https://drive.google.com/uc?export=download&id=13LXboG6W7kXWkN7yTeMZ8PKzwcSUZJR2)


3. Make sure your files arrangment looks like the following
```
R-YOLOv4/
├── train.py
├── test.py
├── detect.py
├── requirements.txt
├── model/
├── tools/
├── outputs/
├── weights
    ├── pretrained/ (for training)
    └── UCAS_AOD/ (for testing and detection)
└── data
    ├── coco.names
    ├── train
        ├── 0
            ├── ...png
            └── ...txt
        └── 1
            ├── ...png
            └── ...txt
    ├── test
        ├── 0
            ├── ...png
            └── ...txt
        └── 1
            ├── ...png
            └── ...txt
    └── detect
        └── ...png
```

### Train

```
usage: train.py [-h] [--data_folder DATA_FOLDER] [--weights_path WEIGHTS_PATH] [--model_name MODEL_NAME]
                [--epochs EPOCHS] [--lr LR] [--batch_size BATCH_SIZE] [--subdivisions SUBDIVISIONS]
                [--img_size IMG_SIZE] [--number_of_classes NUMBER_OF_CLASSES] [--no_augmentation] [--no_mosaic]
                [--no_multiscale] [--custom_dataset]
```

#### Training with custom dataset
1. If you want to train your custom dataset, you can use [labelImg2](https://github.com/chinakook/labelImg2) to help label your data. labelImg2 is capable of labeling rotated objects.
2. Afterwards, convert produced files into txt files and make sure your label format in txt files is the same as [x, y, w, h, angle, label]. An example is given at [here](data/custom_dataset_label_example.txt).
3. Finally, add the --custom_dataset flag when training. For example:
```
python train.py --model_name my_model --custom_dataset
```

#### Training Log
```
---- [Epoch 2/2] ----
+---------------+--------------------+---------------------+---------------------+----------------------+
| Step: 596/600 | loss               | reg_loss            | conf_loss           | cls_loss             |
+---------------+--------------------+---------------------+---------------------+----------------------+
| YoloLayer1    | 0.4302629232406616 | 0.32991039752960205 | 0.09135108441114426 | 0.009001442231237888 |
| YoloLayer2    | 0.7385762333869934 | 0.5682911276817322  | 0.15651139616966248 | 0.013773750513792038 |
| YoloLayer3    | 1.5002599954605103 | 1.1116538047790527  | 0.36262497305870056 | 0.025981156155467033 |
+---------------+--------------------+---------------------+---------------------+----------------------+
Total Loss: 2.669099, Runtime: 404.888372
```

#### Tensorboard
If you would like to use tensorboard for tracking traing process.

* Open additional terminal in the same folder where you are running program.
* Run command ```$ tensorboard --logdir='weights/your_model_name/logs' --port=6006``` 
* Go to [http://localhost:6006/]( http://localhost:6006/)


### Test

| Method | Plane | Car | mAP |
| -------- | -------- | -------- | -------- |
| YOLOv4 (smoothL1-iou) | 97.68 | 90.76 | 94.22|

```
usage: test.py [-h] [--data_folder DATA_FOLDER] [--model_name MODEL_NAME] [--class_path CLASS_PATH]
               [--conf_thres CONF_THRES] [--nms_thres NMS_THRES] [--iou_thres IOU_THRES] [--batch_size BATCH_SIZE]
               [--img_size IMG_SIZE] [--number_of_classes NUMBER_OF_CLASSES] [--custom_dataset]
```

### Detect

```
usage: detect.py [-h] [--data_folder DATA_FOLDER] [--model_name MODEL_NAME] [--class_path CLASS_PATH] [--conf_thres CONF_THRES]
                 [--nms_thres NMS_THRES] [--batch_size BATCH_SIZE] [--img_size IMG_SIZE] [--number_of_classes NUMBER_OF_CLASSES]
```

<img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/P0292.png" alt="car" height="430"/>
<img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/P0259.png" alt="plane" height="413"/>

**Results from other dataset**

<img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/new9_864.jpg" alt="garbage1" height="430"/>
<img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/new9_987.jpg" alt="garbage2" height="430"/>

### References
[yangxue0827/RotationDetection](https://github.com/yangxue0827/RotationDetection)</br>
[eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)</br>
[Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)</br>
[ultralytics/yolov5](https://github.com/ultralytics/yolov5/tree/master/utils)

### TODO

- [x] Mosaic Augmentation


### Credit

**YOLOv4: Optimal Speed and Accuracy of Object Detection**

*Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao*

**Abstract**
There are a huge number of features which are said to improve Convolutional Neural Network (CNN) accuracy. Practical testing of combinations of such features on large datasets, and theoretical justification of the result, is required. Some features operate on certain models exclusively and for certain problems exclusively, or only for small-scale datasets; while some features, such as batch-normalization and residual-connections, are applicable to the majority of models, tasks, and datasets...

```
@article{yolov4,
  title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
  author={Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao},
  journal = {arXiv},
  year={2020}
}
```

**R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object**

*Xue Yang, Junchi Yan, Ziming Feng, Tao He*

**Abstract**
Rotation detection is a challenging task due to the difficulties of locating the multi-angle objects and separating them effectively from the background. Though considerable progress has been made, for practical settings, there still exist challenges for rotating objects with large aspect ratio, dense distribution and category extremely imbalance. In this paper, we propose an end-to-end refined single-stage rotation detector for fast and accurate object detection by using a progressive regression approach from coarse to fine granularity...

```
@article{r3det,
  title={R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object},
  author={Xue Yang, Junchi Yan, Ziming Feng, Tao He},
  journal = {arXiv},
  year={2019}
}
```
