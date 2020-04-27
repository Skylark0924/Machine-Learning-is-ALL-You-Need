# Linear Regression
> Regression

## Principle
作为传统监督学习算法两大类之一，回归的目的就是找到自变量和因变量之间的关系，以便于我们预测（预测温度，预测股票走势）。这种关系中，最简单的就是线性关系。
> 赚钱越多，我越开心 (?)
> 
> 房子越大，房价越高。

作为机器学习的开篇，这里有必要简述一下机器学习的精神：从数据中自适应地归纳、总结、推理(data-based)，以取代人工地规则制定(rule-based)。

![](https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-08-30-taolu.png)

线性关系就是注定了关系曲线方程只能是一次方程，正如我们在小学学过的那样：$y=kx+b$。当然，也可以是多元一次函数: $y=k_1x_1+k_2x_2+\dots+k_nx_n+b$

![](https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-08-30-xianxing.png)

## Procedure
线性回归的求解，就是求方程的系数$k,b$，回忆一下我们在最优化课堂上学到的线性方程组求解方法：**梯度下降法、正规方程法(最小二乘法)**

### 正规方程法
我们将上面的多元一次方程组写成矩阵形式

输入矩阵X为$m\times n+1$的矩阵，即为m个样本，每个样本具有n个特征，多出来的 $1$ 列是为截距b准备的。输出向量Y是m维向量。系数矩阵为 $\theta_{(n+1)\times 1}$，n为k系数，1维截距b。

$$X_{m\times (n+1)}\theta_{(n+1)\times 1}=Y_{m\times 1}\\
X^TX\theta=X^TY\\
\theta = (X^TX)^{-1}X^TY$$

推导见 [用正规方程法求解线性回归](https://zhuanlan.zhihu.com/p/34842727)

```
# 使用正规方程法
X = np.matrix(X)
y = np.matrix(y)
X_T_X = X.T.dot(X)
X_T_X_I_X_T = X_T_X.I.dot(X.T)
X_T_X_I_X_T_X_T_y = X_T_X_I_X_T.dot(y)
self.final_theta = X_T_X_I_X_T_X_T_y
```

### 梯度下降法
同样以矩阵形式开始
1. 随机初始化参数 $\theta$ 
2. 得到估计函数:
    $$h_0(x)=\theta^TX$$
3. 计算损失函数：
   $$J(\theta)=\frac{1}{2} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}$$
4. 按梯度方向下降的方向更新每一个参数，$\alpha$为学习率：
   $$\theta_{i}=\theta_{i}-\alpha \frac{\partial}{\partial \theta} J(\theta)$$
5. 重复以上步骤，直至达到预定精度

![](https://dingyue.ws.126.net/oW096qlMP4GlWxR9TRCXCbEdANt0MPz3cZQfg5AuA3ePP1560695482813.gif)

```
def gradient(self, theta, X, y):
    m = X.shape[0]
    # (m,n).T @ (m, 1) -> (n, 1)，X @ theta等价于X.dot(theta)
    inner = X.T @ (X @ theta - y) + self.regularization.grad(theta)
    return inner / m
    
def batch_gradient_decent(self, theta, X, y, epoch, learning_rate):
    '''
    批量梯度下降, 拟合线性回归, 返回参数和代价
    epoch: 批处理的轮数
    theta: 网络参数
    learning_rate: 学习率
    '''
    cost_data = [self.lr_cost(theta, X, y)]
    _theta = theta.copy()  # 拷贝一份，不和原来的theta混淆

    for _ in range(epoch):
        _theta = _theta - learning_rate * self.gradient(_theta, X, y)
        cost_data.append(self.lr_cost(_theta, X, y))

    return _theta, cost_data

# 使用梯度下降法
final_theta, cost_data = self.batch_gradient_decent(
    self.init_theta, X, y, self.epoch, self.learning_rate)
self.final_theta = final_theta
self.cost = cost_data
```

### 方法对比
![](https://pic2.zhimg.com/80/v2-c973cdee849a4d0a7b92e55c8b520425_1440w.jpg)

### 8 种Python线性回归的方法的速度评测
1. Optimize.curve_fit( )
2. Statsmodels.OLS ( )
3. Scipy.polyfit( ) or numpy.polyfit( )
4. sklearn.linear_model.LinearRegression( )
5. 首先计算x的Moore-Penrose广义伪逆矩阵，然后与y取点积
6. numpy.linalg.lstsq
7. Stats.linregress( )
8. 简单的乘法求矩阵的逆

![](https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-08-30-pingce.png)

相比于我们常用的sklearn.linear_model，直接使用正规矩阵法更快。

## Feature
Advantage:
1. 计算速度快
2. 可以直观地表达变量间的关系

Disadvantage:

1. 不适用于非线性关系

> 不过，可以使用非线性变换在数据预处理过程将特征空间转换为线性空间，再使用线性回归，这和我们后面要提到的SVM的核函数是一个道理。


## Reference 
1. [线性回归 – linear regression](https://easyai.tech/ai-definition/linear-regression/)
2. [机器学习：用正规方程法求解线性回归](https://zhuanlan.zhihu.com/p/34842727)
3. [机器学习：用梯度下降法实现线性回归](https://zhuanlan.zhihu.com/p/33992985)
