# Graph Neural Network (GNN) 实战
> 图神经网络算法族

## Introduction
对于 GNNs 系列算法，强烈推荐参考或直接使用 [dgl (Deep Graph Library) 库](https://www.dgl.ai/)。该库专门用于图神经网络，支持 `pytorch`、`tensorflow` 和 `MXNet` 三种框架，支持 GNN、GCN、Relational-GCN、LGNN、Tree-LSTM、DGMG、JTNN等图神经网络族的算法，代码简练易懂且调用容易。

同时，针对图结构数据，最好熟悉一下 [networkx 库](https://networkx.org/)，者是一个用于创建、操作和研究复杂网络的python软件包。

当然了，**本专栏的重点是 raw python 复现，是自我实现！** 不过，弄懂原理之后，在工程上当然就可以无心理负担地调库了。

## Graph NN
介绍图神经网络的综述在 2018 年之后就如过江之鲫一般，我之前也写过一篇 Graph NN 的归纳文章：

[【归纳综述】Graph Neural Network: An Introduction Ⅰ](https://zhuanlan.zhihu.com/p/158984343 'card')

参考的文章是：

[A Comprehensive Survey on Graph Neural Networks](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1901.00596.pdf) | 2019 Jan

今天要写的 GNN 是最原始版本的，可追溯到 2009 年的 [The graph neural network model](https://persagen.com/files/misc/scarselli2009graph.pdf)，不过这文章的公式看起来不是很友好啊。现在已经很少直接用这个原始版本的 GNN 了，基本都是 GCN 起步了。万幸的是，我找到了一个复现库：

[sailab-code/gnn](https://github.com/sailab-code/gnn 'card')

今天就按照这个代码看一下GNN和普通NN有什么不同。

## GNN



## Reference
1. [Scarselli, Franco, et al. "The graph neural network model." IEEE Transactions on Neural Networks 20.1 (2008): 61-80.](https://persagen.com/files/misc/scarselli2009graph.pdf)
2. 