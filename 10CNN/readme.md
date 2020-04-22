# Convolutional Neural Network
> Image Classification & Feature Extraction

## 
本库使用CIFAR10数据集，只实现了简单的CNN，其余高级variants详见[pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)(包含VGG, Resnet, MobileNet, Googlenet, EfficientNet, Densenet, Shufflenet, Regnet, DPN)

因为 CNN 开始就不再是 sklearn 的范围了，我寻思着加一个tensorflow的原生版本吧。Guess what? 相比于keras和pytorch，这简直是地狱级难度！🙂🙂🙂🙂🙂🙂🙂🙂
拥有keras和pytorch真是件幸运的事。我甚至觉得比我自己写的都麻烦，大家自己对比一下各个类的代码长度吧，都放在[`models.py`](models.py)里了。

- Skylark_CNN
- Keras_CNN
- Torch_CNN
- TF_CNN

