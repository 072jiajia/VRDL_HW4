# VRDL Homework 4
Code for VRDL Homework 4

## Abstract
In this work, I use SRFBN as my model<br>
SRFBN [Paper](https://arxiv.org/pdf/1903.09814.pdf) | [GitHub](https://github.com/Paper99/SRFBN_CVPR19)

## Hardware
The following specs were used to create the solutions.
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
- 2x GeForce RTX 2080 Ti

## Reproducing Submission
To reproduct my submission, do the following steps:
1. [Installation](#installation)
2. [Prepare Data](#dataset-preparation)
3. [Inference](#inference)

## Producing Your Own Submission
To produce your own submission, do the following steps:
1. [Installation](#installation)
2. [Prepare Data](#dataset-preparation)
3. [Train and Make Submission](#train-and-make-prediction)

## Installation
All requirements should be detailed in requirements.txt. Using virtual environment is recommended.
```
virtualenv .
source bin/activate
pip3 install -r requirements.txt
```

## Dataset Preparation
You need to download the [training and testing data](https://drive.google.com/drive/u/0/folders/1H-sIY7zj42Fex1ZjxxSC3PV1pK4Mij6x) by yourself.<br>
And put the data into the same directory as main.py, the directory is structured as:
```
VRDL_HW4
  +- dataset.py
  +- inference.py
  +- main.py
  +- model.py
  +- utils
  +- training_hr_images
  +- testing_lr_images
```

## Inference
Run the following command to reproduct my prediction.
```
python3 inference.py
```
It will generate a directory named "prediction" and my predictions with the same filenames are in it.

## Train and Make Prediction
You can simply run the following command to train your models and make submission.
```
$ python3 main.py
```

The expected training time is:

GPUs| Training Epochs | Training Time
------------- | ------------- | -------------
2x 2080 Ti | 250 | 4 hours

In main.py, run the following code to generate your prediction
```
python3 inference.py
```
It will generate a directory named "prediction" and my predictions with the same filenames are in it.

## Citation
```
@inproceedings{li2019srfbn,
    author = {Li, Zhen and Yang, Jinglei and Liu, Zheng and Yang, Xiaomin and Jeon, Gwanggil and Wu, Wei},
    title = {Feedback Network for Image Super-Resolution},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year= {2019}
}

@inproceedings{wang2018esrgan,
    author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},
    title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
    booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
    year = {2018}
}
```
