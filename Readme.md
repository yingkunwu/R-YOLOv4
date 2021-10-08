# R-YOLOv4

This is a PyTorch-based R-YOLOv4 implementation which combines YOLOv4 model and loss function from R3Det for arbitrary oriented object detection.
(Final project for NCKU INTRODUCTION TO ARTIFICIAL INTELLIGENCE course)

### Introduction
The objective of this project is to provide a capability of oriented object detection for YOLOv4 model. During the model implementation, the most difficult technical challenge that I had ran into is to design a loss function for the model. 

Since I want to add an additional feature for the model, I have to modify the original structure of the model and design a loss function for the modified structure. However, I couldn't find the loss function that is suitable for this structure at first. The result was terrible. The model was either unable to detect objects or unable to predict correct angle of the object. Nevertheless, I tried again and again to address my problem. Every time I failed to get a good result, I reviewed my entire model again, trying to find bugs in my code, and looking for a new plausible loss function by reading lots of paper. 

Finally, I got a successful result by making some adjustments on smooth-L1-IoU loss function proposed by "R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object" to suit my own need. Although the result of my model was not as good as that of others at last, I had acquired a lot of knowledge from the process of solving problem and, more importantly, the result was successful.

### Dataset

**UCAS-High Resolution Aerial Object Detection Dataset (UCAS-AOD)**

Label: x1, y1, x2, y2, x3, y3, x4, y4, theta, x, y, width, height </br>
(x1, y1) is the coordinate located on the upper left of the bounding box, and then (x2, y2), (x3, y3) and (x4, y4) following the clockwise order respectively.

Though it provides theta for each bounding box, it is not within the angle range that I want. You can check out how I calculated the angle that I need in tools/load.py.

### Features

---
#### Loss Function (only for x, y, w, h, theta)

<img src="https://i.imgur.com/zdA9RJj.png" alt="loss" height="90"/>
<img src="https://i.imgur.com/Qi1XFXS.png" alt="angle" height="70"/>

I implemented the loss function proposed by [R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object](https://arxiv.org/abs/1908.05612) and made some adjustments for myself.

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
$ ./setup/setup_train.sh
```
* Or Download it Manually
[yolov4 pretrained weights](https://drive.google.com/uc?export=download&id=1sVD2d_y9VDirA-XOdcVDKCDrQw3e7ZJY) 

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
usage: train.py --model_name YOUR_MODEL_NAME
```

##### Training Log
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

##### Tensorboard
If you would like to use tensorboard for tracking traing process.

* Open additional terminal in the same folder where you are running program.
* Run command ```$ tensorboard --logdir='weights/your_model_name/logs' --port=6006``` 
* Go to [http://localhost:6006/]( http://localhost:6006/)


### Test

| Method | Plane | Car | mAP |
| -------- | -------- | -------- | -------- |
| YOLOv4 (smoothL1-iou) | 97.68 | 90.76 | 94.22|

```
usage: test.py --model_name YOUR_MODEL_NAME
```

### Detect

```
usage: detect.py --model_name YOUR_MODEL_NAME
```

<img src="https://i.imgur.com/UIHJ32m.jpg" alt="car" height="430"/>
<img src="https://i.imgur.com/XzPuOGn.jpg" alt="plane" height="413"/>

### References
[yangxue0827/RotationDetection](https://github.com/yangxue0827/RotationDetection)
[eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
[Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)

### TODO

- [ ] Add weight trained from UCAS-AOD for testing and detection
- [ ] Mosaic Augmentation


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
