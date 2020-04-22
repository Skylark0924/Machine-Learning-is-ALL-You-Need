# K Means
> Clustering

## Principle 
在无监督（无样本标签）情况下的分类，叫做聚类(clustering)。聚类是根据数据之间的关系将数据划分开，如下图，作为一个human，我们可以轻松地区分开 at first glance。

![](https://img-blog.csdn.net/20131226190225921?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvem91eHkwOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## Procedure
1. 预定义聚类的类别数量K
2. 随机选择K个数据作为初始中心点
3. 迭代直至max_iters或中心无变化
   - 计算数据点和所有中心点之间的欧氏距离和
   - 将每个数据点划分到最近的类（最近的中心点）
   - 计算每一类的新中心点（类中数据求平均）

## Reference
1. [K-means Clustering: Algorithm, Applications, Evaluation Methods, and Drawbacks](https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a)
2. [机器学习算法与Python实践之（五）k均值聚类（k-means）](https://blog.csdn.net/zouxy09/article/details/17589329)
