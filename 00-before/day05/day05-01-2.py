# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/6/7 10:46
# @Function: 接day05-01.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

# day05-01.py 绘制了K-means算法使用三个聚类得到的结果
# 该程序显示了使用八个集群将得到什么结果
np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target
estimators = KMeans(n_clusters=8)

fig = plt.figure("fignum", figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

estimators.fit(X)

ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=estimators.labels_.astype(float), edgecolor='k')

# Reorder the labels to have colors matching the cluster results
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.dist = 12
for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
plt.show()


"""
注意：
K_means 算法无法保证聚类结果完全绝对真实的反应实际情况。
首先，选择正确合适的聚类数量不是一件容易的事情
第二，该算法对初始值的设置敏感，容易陷入局部最优。
"""