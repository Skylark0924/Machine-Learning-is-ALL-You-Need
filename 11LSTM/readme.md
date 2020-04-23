# Long Short-Term Memory (LSTM)
> Sequential Information Processing 

**Remember! Please Download the [Penn TreeBank (PTB)](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz) dataset manually first! And copy it into ./dataset directory. Never mind. I have done it.**

## Principle 
RNN尤其是LSTM的提出是具有创造性价值的，相比于NN的函数拟合能力，CNN的视觉特征提取能力，RNN更着眼于理解和记忆。如果说CNN是视觉神经，那么RNN就是脑前庭上皮组织。

![](https://pic3.zhimg.com/v2-bd1f88ed4748a7f5f347287f07e9292a_1200x500.jpg)

具体的公式就这些：
$$\begin{array}{l}
i=\sigma\left(W_{i i} x+b_{i i}+W_{h i} h+b_{h i}\right) \\
f=\sigma\left(W_{i f} x+b_{i f}+W_{h f} h+b_{h f}\right) \\
g=\tanh \left(W_{i g} x+b_{i g}+W_{h g} h+b_{h g}\right) \\
o=\sigma\left(W_{i o} x+b_{i o}+W_{h o} h+b_{h o}\right) \\
c^{\prime}=f * c+i * g \\
h^{\prime}=o * \tanh \left(c^{\prime}\right)
\end{array}$$
简单来说就是，LSTM一共有三个门，输入门，遗忘门，输出门。

![](https://miro.medium.com/max/1400/1*yBXV9o5q7L_CvY7quJt3WQ.png)


### Forget gate
![](https://miro.medium.com/max/1400/1*GjehOa513_BgpDDP6Vkw2Q.gif)

This gate 决定了那些info需要被遗弃，哪些要保留。
- 上一cell的隐状态$h_{i-1}$与本cell的输入$x_i$连在一起
- 经过sigmoid
- 靠近0的被遗忘，靠近1的本保留。
- 得到向量 $f_i$


### Input gate
![](https://miro.medium.com/max/1400/1*TTmYy7Sy8uUXxUXfzmoKbA.gif)

The input gate decides what information is relevant to add from the current step.

- $h_{i-1}+x_i$ 经过一个sigmoid，来确定要更新那些值，0-1不重要，1表示重要
- 同时，$h_{i-1}+x_i$经过tanh函数，用以将值压缩到(-1,1)区间
- 将二者的结果乘到一起，sigmoid的输出将决定来保留tanh输出的哪些info

### Cell State
![](https://miro.medium.com/max/1400/1*S0rXIeO_VoUVOyrYHckUWg.gif)

- 上一cell state $c_{i-1}$ 的乘以遗忘门输出 $f_i$，部分info被遗弃了
- 加上输入门的输出$i_i$ 就得到本cell state


### Output gate 
![](https://miro.medium.com/max/1400/1*VOXRGhOShoWWks6ouoDN3Q.gif)

输出门决定了下一个隐状态，这个隐状态是包含过往信息的，同时也被用于预测。

- $h_{i-1}+x_i$ 经过sigmoid，这和遗忘门一样
- cell state经过tanh的范围缩放
- 两者相乘，用以筛选cell state中的重要信息，将其保留在隐状态中，用于下一个cell的计算

## Reference
1. [Keras RNN](https://keras.io/zh/layers/recurrent/)
2. [Keras LSTM tutorial – How to easily build a powerful deep learning language model](https://adventuresinmachinelearning.com/keras-lstm-tutorial/)
3. [Illustrated Guide to LSTM’s and GRU’s: A step by step explanation](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
4. [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
5. [LSTM Neural Network from Scratch Kaggle](https://www.kaggle.com/navjindervirdee/lstm-neural-network-from-scratch)