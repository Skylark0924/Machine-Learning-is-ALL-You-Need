# VGG

## Introduction

VGG stands for Visual Geometry Group. It is a standard deep Convolutional Neural Network
(CNN) architecture with multiple layers. The “deep” refers to the number of layers with VGG-16 or VGG-19 consisting of 16 and 19 convolutional layers.

The VGG architecture is the basis of ground-breaking object recognition models. Developed as
a deep neural network, the VGGNet also surpasses baselines on many tasks and datasets beyond
ImageNet. Moreover, it is now still one of the most popular image recognition architectures. The network architecture of VGG variants are shown as follows:


![](img/image-20220317155555066.png)


## Train on CIFAR-10 dataset
The dataset will download itself.

The code can be run by 
```
cd ./CV_VGG
python train.py
```