# Boosting
> Ensemble Model

## Principle
提升方法（Boosting）是一种可以用来减小监督学习中偏差的机器学习算法。主要也是学习一系列弱分类器，并将其组合为一个强分类器。

之前的Random Forest属于Bagging，Boosting类集成算法有以下不同：
- Bagging中每个训练集互不相关，也就是每个基分类器互不相关，而Boosting中训练集要在上一轮的结果上进行调整，也使得其不能并行计算
- Bagging中预测函数是均匀平等的，但在Boosting中预测函数是加权的

Bagging: 

![](https://images2015.cnblogs.com/blog/1042406/201612/1042406-20161204200000787-1988863729.png)

Boosting:

![](https://images2015.cnblogs.com/blog/1042406/201612/1042406-20161204194331365-2142863547.png)

Boosting算法的工作机制:
- 从训练集用初始权重训练出一个弱学习器1
- 根据弱学习的学习误差率表现来更新训练样本的权重，使得之前弱学习器1学习误差率高的训练样本点的权重变高，使得这些误差率高的点在后面的弱学习器2中得到更多的重视。
- 基于调整权重后的训练集来训练弱学习器2
- 如此重复进行，直到弱学习器数达到事先指定的数目T
- 最终将这T个弱学习器通过集合策略进行整合，得到最终的强学习器。　

那么问题来了：
- 如何计算学习误差率e?
- 如何得到弱学习器权重系数α?
- 如何更新样本权重D?
- 使用何种结合策略？

这就是不同的Boosting算法要解决的。

### AdaBoost 分类
输入为样本集T={(x,y1),(x2,y2),...(xm,ym)}，输出为{-1, +1}，弱分类器算法, 弱分类器迭代次数K。

输出为最终的强分类器f(x)

1. 初始化样本集权重为
   $$D(1)=\left(w_{11}, w_{12}, \ldots w_{1 m}\right) ; \quad w_{1 i}=\frac{1}{m} ; \quad i=1,2 \ldots m$$
2. 对于$k=1,2, ..., K$:
   - 使用具有权重$D(k)$的样本集来训练数据，得到弱分类器$G_k(x)$
   - 计算$G_k(x)$的分类误差率
        $$e_{k}=P\left(G_{k}\left(x_{i}\right) \neq y_{i}\right)=\sum_{i=1}^{m} w_{k i} I\left(G_{k}\left(x_{i}\right) \neq y_{i}\right)$$
   - 计算弱分类器的系数，**误差率小的弱分类器权重系数越大**
        $$\alpha_{k}=\frac{1}{2} \log \frac{1-e_{k}}{e_{k}}$$
   -  更新样本集的权重分布，**如果第i个样本分类错误，则$y_iG_k(x_i)<0$，导致样本的权重在第k+1个弱分类器中增大，如果分类正确，则权重在第k+1个弱分类器中减少**
        $$w_{k+1, i}=\frac{w_{k i}}{Z_{K}} \exp \left(-\alpha_{k} y_{i} G_{k}\left(x_{i}\right)\right) \quad i=1,2, \ldots m$$
    这里$Z_k$是规范化因子
    $$Z_{k}=\sum_{i=1}^{m} w_{k i} \exp \left(-\alpha_{k} y_{i} G_{k}\left(x_{i}\right)\right)$$
3. 构建最终分类器为：
     $$f(x)=\operatorname{sign}\left(\sum_{k=1}^{K} \alpha_{k} G_{k}(x)\right)$$

> Tips:   
> x>0，sign(x)=1;
> x=0，sign(x)=0;
> x<0， sign(x)=-1；

对于Adaboost多元分类，最主要区别在弱分类器的系数上。比如Adaboost SAMME算法，它的弱分类器的系数：
$$\alpha_{k}=\frac{1}{2} \log \frac{1-e_{k}}{e_{k}}+\log (R-1)$$
其中R为类别数。从上式可以看出，如果是二元分类，R=2，则上式和我们的二元分类算法中的弱分类器的系数一致。

> 为什么Adaboost的弱学习器权重系数公式和样本权重更新公式是这个样子的？其实这是从损失函数里推导出来。详见[链接](https://www.cnblogs.com/pinard/p/6133937.html)第三章。


#### Feature
优点：
1. Adaboost作为分类器时，分类精度很高
2. 在Adaboost的框架下，可以使用各种回归分类模型来构建弱学习器，非常灵活。
3. 作为简单的二元分类器时，构造简单，结果可理解。
4. 不容易发生过拟合

缺点：

对异常样本敏感，异常样本在迭代中可能会获得较高的权重，影响最终的强学习器的预测准确性。

### Gradient Boost Decision Tree
Adaboost是利用前一轮迭代弱学习器的误差率来更新训练集的权重，这样一轮轮的迭代下去。GBDT也是迭代，使用了前向分布算法，但是弱学习器限定了只能使用CART回归树模型，同时迭代思路和Adaboost也有所不同。

Adaboost的对于强学习器$f_{t}(x)$的损失函数就是$L(y,f_t(x))$，而GBDT的损失函数$L(y,f_t(x)=L(y,f_{t-1}(x)+h_t(x)))$，目标就是寻找在本轮寻找一个弱学习器$h_t(x)$，要让损失尽量小。

> 一个通俗的例子解释，假如有个人30岁，我们首先用20岁去拟合，发现损失有10岁，这时我们用6岁去拟合剩下的损失，发现差距还有4岁，第三轮我们用3岁拟合剩下的差距，差距就只有一岁了。如果我们的迭代轮数还没有完，可以继续迭代下面，每一轮迭代，拟合的岁数误差都会减小。

那么问题来了，如何拟合这种损失？

#### GBDT的负梯度拟合
针对这个问题，大牛Freidman提出了**用损失函数的负梯度来拟合本轮损失的近似值，进而拟合一个CART树**。第t轮的第i个样本的损失函数的负梯度表示为

$$r_{t i}=-\left[\frac{\left.\partial L\left(y_{i}, f\left(x_{i}\right)\right)\right)}{\partial f\left(x_{i}\right)}\right]_{f(x)=f_{t-1}}$$

### Xgboost






## Reference
1. [集成学习（Ensemble Learning)](https://zhuanlan.zhihu.com/p/27689464)
2. [集成学习之Adaboost算法原理小结](https://www.cnblogs.com/pinard/p/6133937.html)
3. [梯度提升树(GBDT)原理小结](https://www.cnblogs.com/pinard/p/6140514.html)
4. [XGBoost算法原理小结](https://www.cnblogs.com/pinard/p/10979808.html)
