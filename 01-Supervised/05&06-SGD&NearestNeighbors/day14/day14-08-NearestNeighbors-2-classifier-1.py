# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/12 14:20
# @Function: 最近邻分类 https://www.scikitlearn.com.cn/0.21.3/7/#162
"""
最近邻分类属于 基于实例的学习 或 非泛化学习 ：它不会去构造一个泛化的内部模型，而是简单地存储训练数据的实例。
两种最近邻分类器
    1. 基于每个查询点的k个最近邻实现，其中k是用户指定的整数值。
    2. RadiusNeighborsClassifier 基于每个查询点的固定半径 r 内的邻居数量实现， 其中 r 是用户指定的浮点数值。
        数据不均匀时效果更好，高维数据低效。

"""

# 示例：

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

h = .02

cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ['darkorange', 'c', 'darkblue']
# 默认值 weights = 'uniform' 为每个近邻分配统一的权重。而 weights = 'distance' 分配权重与查询点的距离成反比。
for weights in ['uniform', 'distance']:
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y], palette=cmap_bold, alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])

plt.show()


"""
对K近邻分类的理解：
假设有6个样本点  点1,点2,点3为类1  点4,点5,点6为类0
有一个预测点7
通过欧式距离(或其他)计算出6个样本点与预测点的距离。假设距离从小到大排序为1,2,4,5,6,3
当K=3时，选择距离预测点最近的三个点，即1,2,4，这三个点对应的类别是1  1  0，则预测点7为类别1
当K=4时，选择距离预测点最近的四个点，即1,2,4,5，这三个点对应的类别是1  1  0  0，则需要重新选择K值
当K=5时，选择距离预测点最近的五个点，即1,2,4,5,6，这三个点对应的类别是1  1  0  0  0，则预测点7为类别0

若选择较小的K值，近似误差会较小，而估计误差会增大。易发生过拟合现象
若选择较大的K值，估计误差减小，近似误差增大。
通常采用交叉验证法来选取最优的K值


"""

