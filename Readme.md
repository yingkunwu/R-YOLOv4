# R-YOLOv4

### Introduction
The objective of this project is to adapt YOLOv4 model to detecting oriented objects. As a result, modifying the original loss function of the model is required. I got a successful result by increasing the number of anchor boxes with different rotating angle and combining smooth-L1-IoU loss function proposed by [R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object](https://arxiv.org/abs/1908.05612) into the original loss for bounding boxes.

### Features

---
#### Loss Function (only for x, y, w, h, theta)

<img src="https://i.imgur.com/zdA9RJj.png" alt="loss" height="90"/>
<img src="https://i.imgur.com/Qi1XFXS.png" alt="angle" height="70"/>

---

### Setup

1. Clone repository
    ```
    $ git clone https://github.com/kunnnnethan/R-YOLOv4.git
    $ cd R-YOLOv4/
    ```
2. Create Conda Environment
    ```
    $ conda create -n ryolo python=3.8
    $ conda activate ryolo
    ```
3. Install PyTorch and torchvision following the [official instructions](https://pytorch.org), e.g.,
    ```
    If you are using CUDA 11.8 version
    $ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
4. Install required libraries
    ```
    $ pip install -r requirements.txt
    ```
5. Install detectron2 for calculating SkewIoU on GPU following the [official instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html), e.g.,
    ```
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    ```

6. Download  pretrained weights</br>
    [weights](https://drive.google.com/uc?export=download&id=1A_C4KzYsSa8yidp_5Wf9B9DIwPG7N7C8)
    
7. Make sure your files arrangment looks like the following</br>
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
    
### Train

I have implemented methods to load and train three different datasets. They are UCAS-AOD, DOTA, and custom dataset respectively. You can check out how I loaded those dataset into the model at [/datasets](https://github.com/kunnnnethan/R-YOLOv4/tree/main/datasets). The angle of each bounding box is limited in `(- pi/2,  pi/2]`, and the height of each bounding box is always longer than it's width.

```
$ python train.py --data data/UCAS_AOD.yaml --hyp data/hyp.yaml --model_name ryolov4 --batch_size 16 --img_size 608
```

You can run [display_inputs.py](https://github.com/kunnnnethan/R-YOLOv4/blob/main/display_inputs.py) to visualize whether your data is loaded successfully.

#### UCAS-AOD dataset

Please refer to [this repository](https://github.com/kunnnnethan/UCAS-AOD-benchmark) to rearrange files so that it can be loaded and trained by this model.</br>
You can download the [weight](https://drive.google.com/uc?export=download&id=1h-OKkkPAxPS9PvMjOU03T6KZdHUaDY5G) that I trained from UCAS-AOD.

#### DOTA dataset

Download the official dataset from [here](https://captain-whu.github.io/DOTA/dataset.html). The original files should be able to be loaded and trained by this model.</br>
You can download the [weight](https://drive.google.com/uc?export=download&id=19xET9cnpPbp5fvSkLY4NiUvKGQNyVPiB) that I trained from DOTA.

#### Train with custom dataset
1. Use [labelImg2](https://github.com/chinakook/labelImg2) to help label your data. labelImg2 is capable of labeling rotated objects.
2. Move your data folder into the `R-YOLOv4/data` folder.
3. Run xml2txt.py
    1. generate txt files:
    ```python xml2txt.py --data_folder your-path --action gen_txt```
    2. delete xml files:
    ```python xml2txt.py --data_folder your-path --action del_xml```

A [trash](https://drive.google.com/uc?export=download&id=1YBDtCoRXEVkPQUUcfoChWKq8WVzm7IF-) custom dataset that I made and the [weight](https://drive.google.com/uc?export=download&id=1UpiBurcQr52ZDSjZZDzO6XsUzkWEwJ__) trained from it are provided for your convenience.

### Test
```
python test.py --data data/UCAS_AOD.yaml --hyp data/hyp.yaml --weight_path weights/ryolov4/best.pth --batch_size 8 --img_size 608
```

### detect
```
python detect.py --data data/UCAS_AOD.yaml --weight_path weights/ryolov4/best.pth --batch_size 8 --img_size 608
```

#### Tensorboard
If you would like to use tensorboard for tracking traing process.

* Open additional terminal in the same folder where you are running program.
* Run command ```$ tensorboard --logdir=weights --port=6006``` 
* Go to [http://localhost:6006/]( http://localhost:6006/)

### Results

#### UCAS_AOD

| Method | Plane | Car | mAP |
| -------- | -------- | -------- | -------- |
| YOLOv4 (smoothL1-iou) | 98.2 | 82.4 | 90.03|

<img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/UCAS_AOD/P0292.png" alt="car" height="430"/>
<img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/UCAS_AOD/P0769.png" alt="plane" height="413"/>

#### DOTA

DOTA have not been tested yet. (It's quite difficult to test because of large resolution of images)
<img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/DOTA/P0006.png" alt="DOTA" height="430"/><img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/DOTA/P0031.png" alt="DOTA" height="430"/>

#### trash (custom dataset)

| Method | Tetra Pak | Aluminum Can | mAP |
| -------- | -------- | -------- | -------- |
| YOLOv4 (smoothL1-iou) | 100.00 | 100.00 | 100.00|

<img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/trash/480.jpg" alt="garbage1" height="410"/>
<img src="https://github.com/kunnnnethan/R-YOLOv4/blob/main/outputs/trash/540.jpg" alt="garbage2" height="410"/>


### References

[WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7/tree/main)</br>
[ultralytics/yolov5](https://github.com/ultralytics/yolov5/tree/master/utils)</br>
[Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)</br>
[yangxue0827/RotationDetection](https://github.com/yangxue0827/RotationDetection)</br>
[eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)</br>

**YOLOv4: Optimal Speed and Accuracy of Object Detection**

```
@article{yolov4,
  title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
  author={Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao},
  journal = {arXiv},
  year={2020}
}
```

**R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object**

```
@article{r3det,
  title={R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object},
  author={Xue Yang, Junchi Yan, Ziming Feng, Tao He},
  journal = {arXiv},
  year={2019}
}
```
