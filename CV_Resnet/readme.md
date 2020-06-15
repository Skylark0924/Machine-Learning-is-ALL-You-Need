# Resnet 
> Computer Vision

2015 | [Paper](https://arxiv.org/abs/1512.03385) | [Code]()

## Principle
### Target
Resnet 用来解决**深度网络的退化问题**，即给网络叠加更多的层后，性能却快速下降的情况。这些内容我就不抄了，详见reference。

要想改进退化问题，可以从两个方面入手：
- 更好的初始化、更好的梯度下降算法等；
- 调整模型结构，让模型更易于优化。

### Residual learning
ResNet的作者从后者入手，探求更好的模型结构。将堆叠的几层layer称之为一个block，对于某个block，其可以拟合的函数为$F(x)$，如果期望的潜在映射为$H(x)$，与其让$F(x)$ 直接学习潜在的映射，不如去学习残差$H(x)−x$，即

$$F(x):=H(x)−x$$

这样原本的前向路径上就变成了$F(x)+x$，用$F(x)+x$来拟合$H(x)$。作者认为这样可能更易于优化，因为相比于让$F(x)$学习成恒等映射，让$F(x)$学习成0要更加容易——后者通过L2正则就可以轻松实现。这样，**对于冗余的block，只需$F(x)→0$就可以得到恒等映射，性能不减**。这样就确保了加层不至于性能反而更差。

![](https://pic4.zhimg.com/80/v2-252e6d9979a2a91c2d3033b9b73eb69f_720w.jpg)

为什么残差学习相对更容易，从直观上看残差学习需要学习的内容少，因为残差一般会比较小，学习难度小点。不过我们可以从数学的角度来分析这个问题，首先残差单元可以表示为：

$$
H(x_l)=x_l+F(x_l,W_l)\\
x_{l+1}=\sigma(H(x_l))
$$

基于上式，我们先忽略激活层的影响，求得从浅层 $l$ 到深层 $L$ 的学习特征为：

$$x_L=x_l+\sum^{L-1}_{i=l}F(x_i,W_i)$$

利用链式规则，可以求得反向过程的梯度：

$$\frac{\partial \operatorname{loss}}{\partial x_{l}}=\frac{\partial \operatorname{loss}}{\partial x_{L}} \cdot \frac{\partial x_{L}}{\partial x_{l}}=\frac{\partial \operatorname{loss}}{\partial x_{L}} \cdot\left(1+\frac{\partial}{\partial x_{L}} \sum_{i=l}^{L-1} F\left(x_{i}, W_{i}\right)\right)$$

式子的第一个因子 $\frac{\partial \operatorname{loss}}{\partial x_{L}}$ 表示的损失函数到达 L 的梯度，小括号中的1表明短路机制可以无损地传播梯度，而另外一项残差梯度则需要经过带有weights的层，梯度不是直接传递过来的。残差梯度不会那么巧全为-1，而且就算其比较小，有1的存在也不会导致梯度消失。所以残差学习会更容易。要注意上面的推导并不是严格的证明。

### Resnet Structure
#### Residual Block
上文提到残差块是多隔层的组合，上文给出的示意图只是一个简化原理图，需要确定的还有很多。

- 残差路径如何设计？（H(x)怎么设计）
- shortcut路径如何设计？（恒等映射路径是否可以有其他形式）
- Residual Block之间怎么连接？

**残差路径可以大致分成2种：**

- bottleneck block结构，即下图右中的1×1 卷积层，用于**先降维再升维**，主要出于**降低计算复杂度**的现实考虑；
- basic block结构，即没有瓶颈构型的。

![](https://s2.ax1x.com/2020/02/21/3K34c8.png)

**shortcut路径设计分为两种：**

- 恒等映射；
- 需要经过1×1卷积来升维 or/and 降采样，主要作用是将输出与F(x)路径的输出保持shape一致。

![](https://s2.ax1x.com/2020/02/23/3l4cD0.png)

**Residual Block之间的衔接：**

F(x)+x经过ReLU后直接作为下一个block的输入x

#### Network
直观看一下ResNet-34与34-layer plain net和VGG的对比，以及堆叠不同数量Residual Block得到的不同ResNet。

![](https://s2.ax1x.com/2020/02/21/3u8Wwj.png)

![](https://pytorch.org/assets/images/resnet.png)

其特点很明显：
- 与plain net相比，ResNet多了很多“旁路”，即shortcut路径，其首尾圈出的layers构成一个Residual Block；
- ResNet中，所有的Residual Block都没有pooling层，**降采样是通过conv的stride实现的**；
- 分别在conv3_1、conv4_1和conv5_1 Residual Block，降采样1倍，同时feature map数量增加1倍，如图中虚线划定的block；
- **通过Average Pooling得到最终的特征**，而不是通过全连接层；
- 每个卷积层之后都紧接着BatchNorm layer，为了简化，图中并没有标出；


``` 
def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

```



## Reference
1. [你必须要知道CNN模型：ResNet](https://zhuanlan.zhihu.com/p/31852747)
2. [详解残差网络](https://zhuanlan.zhihu.com/p/42706477)
3. [ResNet详解与分析](https://www.cnblogs.com/shine-lee/p/12363488.html)