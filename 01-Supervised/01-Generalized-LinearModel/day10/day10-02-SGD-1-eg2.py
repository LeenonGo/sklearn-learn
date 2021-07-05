# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/5 11:16
# @Function: 使用类SGDClassifier实现了一个简单的随机梯度下降学习示例
# 该示例支持不同的损失函数和分类惩罚。
# 下面是SGDClassifier经过损失训练的决策边界 ，相当于线性 SVM。
# 使用SGD训练的线性支持向量机分类器绘制两类可分离数据集中的最大边距分离超平面。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_blobs

X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=200)

clf.fit(X, Y)

xx = np.linspace(-1, 5, 10)
yy = np.linspace(-1, 5, 10)

X1, X2 = np.meshgrid(xx, yy)
Z = np.empty(X1.shape)
for (i, j), val in np.ndenumerate(X1):
    x1 = val
    x2 = X2[i, j]
    p = clf.decision_function([[x1, x2]])
    Z[i, j] = p[0]
levels = [-1.0, 0.0, 1.0]
linestyles = ['dashed', 'solid', 'dashed']
colors = 'k'
plt.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolor='black', s=20)

plt.axis('tight')
plt.show()

