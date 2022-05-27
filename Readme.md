# R-YOLOv4

This is a PyTorch-based R-YOLOv4 implementation which combines YOLOv4 model and loss function from R3Det for arbitrary oriented object detection.
(Final project for NCKU INTRODUCTION TO ARTIFICIAL INTELLIGENCE course)

### Introduction
The objective of this project is to adapt YOLOv4 model to detecting oriented objects. As a result, modifying the original loss function of the model is required. I got a successful result by increasing the number of anchor boxes with different rotating angle and combining smooth-L1-IoU loss function proposed by [R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object](https://arxiv.org/abs/1908.05612) into the original loss for bounding boxes.

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

---

### Usage

1. Clone and Setup Environment
    ```
    $ git clone https://github.com/kunnnnethan/R-YOLOv4.git
    $ cd R-YOLOv4/
    ```
    Create Conda Environment
    ```
    $ conda env create -f environment.yml
    ```
    Create Python Virtual Environment
    ```
    $ python3.8 -m venv (your environment name)
    $ source ~/your-environment-name/bin/activate
    $ pip3 install torch torchvision torchaudio
    $ pip install -r requirements.txt
    ```

2. Download  pretrained weights</br>
    [weights](https://drive.google.com/uc?export=download&id=1zPSXWwbmNwUV4OHFuKoILByx_hetPBXM)
    
3. Make sure your files arrangment looks like the following</br>
    Note that each of your dataset folder in `data` should split into three files, namely `train`, `test`, and `detect`.
    ```
    R-YOLOv4/
    ├── train.py
    ├── test.py
    ├── detect.py
    ├── xml2txt.py
    ├── environment.xml
    ├── requirements.txt
    ├── model/
    ├── datasets/
    ├── lib/
    ├── outputs/
    ├── weights/
        ├── pretrained/ (for training)
        └── UCAS-AOD/ (for testing and detection)
    └── data/
        └── UCAS-AOD/
            ├── class.names
            ├── train/
                ├── ...png
                └── ...txt
            ├── test/
                ├── ...png
                └── ...txt
            └── detect/
                └── ...png
    ```
4. Train, Test, and Detect</br>
    Please refer to `lib/options.py` to check out all the arguments.
    
### Train

I have implemented methods to load and train three different datasets. They are UCAS-AOD, DOTA, and custom dataset respectively. You can check out how I loaded those dataset into the model at [/datasets](https://github.com/kunnnnethan/R-YOLOv4/tree/main/datasets). The angle of each bounding box is limited in `(- pi/2,  pi/2]`, and the height of each bounding box is always longer than it's width.

You can run [display_inputs.py](https://github.com/kunnnnethan/R-YOLOv4/blob/main/display_inputs.py) to visualize whether your data is loaded successfully.

#### UCAS-AOD dataset

Please refer to [this repository](https://github.com/kunnnnethan/UCAS-AOD-benchmark) to rearrange files so that it can be loaded and trained by this model.</br>
You can download the [weight](https://drive.google.com/uc?export=download&id=1UlewA9dcXsCiCbuvKCU6vL8AqSYeI3mj) that I trained from UCAS-AOD.
```
While training, please specify which dataset you are using.
$ python train.py --dataset UCAS_AOD
```

#### DOTA dataset

Download the official dataset from [here](https://captain-whu.github.io/DOTA/dataset.html). The original files should be able to be loaded and trained by this model.
```
While training, please specify which dataset you are using.
$ python train.py --dataset DOTA
```

#### Train with custom dataset
1. Use [labelImg2](https://github.com/chinakook/labelImg2) to help label your data. labelImg2 is capable of labeling rotated objects.
2. Move your data folder into the `R-YOLOv4/data` folder.
3. Run xml2txt.py
    1. generate txt files:
    ```python xml2txt.py --data_folder your-path --action gen_txt```
    2. delete xml files:
    ```python xml2txt.py --data_folder your-path --action del_xml```
    
A [trash](https://drive.google.com/uc?export=download&id=1jZHgezUkKExLjSXpALd3N8lXX5kvH_S-) custom dataset that I made and the [weight](https://drive.google.com/uc?export=download&id=1ppR__Un4NRHA8BDwpd9Qz7NAfh4fchwM) trained from it are provided for your convenience.
```
While training, please specify which dataset you are using.
$ python train.py --dataset custom
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

### Results

#### UCAS_AOD

| Method | Plane | Car | mAP |
| -------- | -------- | -------- | -------- |
| YOLOv4 (smoothL1-iou) | 98.05 | 92.05 | 95.05|

<img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/UCAS_AOD/P0292.png" alt="car" height="430"/>
<img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/UCAS_AOD/P0769.png" alt="plane" height="413"/>

#### DOTA

DOTA have not been tested yet. (It's quite difficult to test because of large resolution of images)
<img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/DOTA/P0006.png" alt="DOTA" height="430"/><img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/DOTA/P0031.png" alt="DOTA" height="430"/>

#### trash (custom dataset)

| Method | Plane | Car | mAP |
| -------- | -------- | -------- | -------- |
| YOLOv4 (smoothL1-iou) | 100.00 | 100.00 | 100.00|

<img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/trash/480.jpg" alt="garbage1" height="410"/>
<img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/trash/540.jpg" alt="garbage2" height="410"/>


### TODO

- [x] Mosaic Augmentation
- [x] Mixup Augmentation


### References

[yangxue0827/RotationDetection](https://github.com/yangxue0827/RotationDetection)</br>
[eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)</br>
[Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)</br>
[ultralytics/yolov5](https://github.com/ultralytics/yolov5/tree/master/utils)

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
