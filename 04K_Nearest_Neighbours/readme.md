# K-近邻算法 (KNN)
> 分类&回归算法

### 原理
KNN是通过测量不同特征值之间的距离进行分类。即从已有数据中找出简单的规律。

**思路:**
如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别，其中K通常是不大于20的整数。

![Best Knn GIFs | Gfycat](https://thumbs.gfycat.com/WildSorrowfulChevrotain-max-1mb.gif)

**条件:**

1. 所有邻居已被正确分类

**距离测定:**

1. 欧式距离: $d(x,y)=\sqrt{\sum^n_{k=1}(x_k-y_k)^2}$
2. 曼哈顿距离: $d(x,y)=\sqrt{\sum^n_{k=1}|x_k-y_k|}$



### 步骤

1. 计算测试数据与各个训练数据之间的距离；
2. 按照距离的递增关系进行排序；
3. 选取距离最小的K个点；
4. 确定前K个点所在类别的出现频率；
5. 返回前K个点中出现频率最高的类别作为测试数据的预测分类。

```python
def predict(self, X_test, X_train, y_train):
    y_predict = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
    	# 测试的数据和训练的各个数据的欧式距离
        distances = np.zeros((X_train.shape[0], 2))  
        for j in range(X_train.shape[0]):
        	# 计算欧式距离
            dis = self.euclidean_distance(X_test[i], X_train[j])
            # 测试集到的每个训练集的数据的分类标签
            label = y_train[j]  
            distances[j] = [dis, label]
            # argsort()得到测试集到训练的各个数据的欧式距离从小到大排列并且得到序列
            # 然后再取前k个.
            k_nearest_neighbors = distances[distances[:, 0].argsort(
            )][:self.k_neighbors]

            # 利用np.bincount统计k个近邻里面各类别出现的次数
            counts = np.bincount(k_nearest_neighbors[:, 1].astype('int'))

            # 得出每个测试数据k个近邻里面各类别出现的次数最多的类别
            testLabel = counts.argmax()
            y_predict[i] = testLabel

    return np.array(y_predict)

def euclidean_distance(self, x1, x2):
    """ Calculates the l2 distance between two vectors """
    distance = 0
    # Squared distance between each coordinate
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)
```




### 特性
**优点:**

1. 简单好用，既可以用来做分类也可以用来做回归；
2. 可用于数值型数据和离散型数据；
3. 训练时间复杂度为O(n)；无数据输入假定；
4. 对异常值不敏感。

**缺点:**

1. 计算复杂性高；空间复杂性高；
2. 存在样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）；
3. 数据量不能过大也不能过小
4. **无法给出数据的内在含义**。