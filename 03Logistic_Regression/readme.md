# Logistic Regression
> Classification

## Principle
逻辑回归并不是用于回归，而是用于解决二分类问题，输出的结果表达了某件事情发生的**可能性**(不等价于数学意义上的概率，因为其不存在。之所以名字里包含回归，是因为其建立在线性回归之上，引入了非线性Sigmoid函数 $g(z)$ 。函数图像如下：

![](https://pic2.zhimg.com/80/v2-1562a80cf766ecfe77155fa84931e745_1440w.png)

线形回归：$h_{\theta}(x)=\theta^{T} x$

逻辑回归：$h_{\theta}(x)=g\left(\theta^{T} x\right), g(z)=\frac{1}{1+e^{-z}}$，
即$h_{\theta}(x)=\frac{1}{1+e^{-\theta^{T} x}}$

## Procedure
逻辑回归实现起来和[上一章](../01Single_Linear_Regression/readme.md)的梯度下降法一样，只不过多了一步sigmoid。
```
# 梯度训练n_iterations轮
for i in range(self.epoch):
    h_x = X.dot(self.init_theta)
    y_pred = sigmoid(h_x)
    theta_grad = X.T.dot(y_pred - y)
    self.init_theta -= self.learning_rate * theta_grad
```

## Feature
Advantages：
1. 简单又快

Disadvantages:
1. 容易欠拟合，一般准确度不太高；
2. 不能很好地处理大量多类特征或变量；
3. 对于非线性特征，需要进行转换；
4. 适用于二分类问题

> 使用softmax激活函数就可以扩展到多分类

## Reference
1. [逻辑回归（Logistic Regression）（一）](https://zhuanlan.zhihu.com/p/28408516)