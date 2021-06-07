# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/6/7 10:14
# @Function: 聚类: 对样本数据进行分组
# 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets
"""
可以利用聚类解决的问题：
对于一个样本，知道有几种不同的类型，但是不知道每一个样本具体属于哪一种类型。

聚类：将样本进行分组，相似的样本被聚在一起，而不同组别之间的样本是有明显区别的。

"""

"""
关于聚类有很多不同的聚类标准和相关算法
其中最简便的算法是 K-means
"""

np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target


# 初始化不同的估计器
estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
              ('k_means_iris_3', KMeans(n_clusters=3)),
              ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1,
                                               init='random'))]

fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
for name, est in estimators:
    fig = plt.figure(name, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)  # 训练样本
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=labels.astype(float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    # ax.set_title(titles[fignum - 1])
    ax.set_title(name)
    ax.dist = 12
    fignum = fignum + 1

# 图中首先显示了K-means算法使用三个聚类得到的结果。
# 然后显示了错误初始化对分类过程的影响：
# 通过将n_init仅设置为1（默认值为10），可以减少使用不同质心种子运行算法的次数.
# n_init就是初始化的次数
# 因为k means无法达到全局最优，每次收敛到最后的结果是局部最优，
# 所以就需要跑很多次独立初始化的k-means，比如说n_init=10就是跑10次。然后从这10次里面选最优的那个。
# 不同初始化只取决于不同的初始点，也就是初始质心。


# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist = 12

plt.show()


