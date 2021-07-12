# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/12 16:02
# @Function: 邻域成分分析 NCA, NeighborhoodComponentsAnalysis
# https://www.scikitlearn.com.cn/0.21.3/7/#166
"""
距离度量学习算法
目的是提高最近邻分类相对于标准欧氏距离的准确性
拟合一个尺寸为(n_components, n_features)的最优线性变换矩阵，使所有被正确分类的概率样本的和最大
本相似度计算方法基于马氏距离
既属于度量学习范畴，又是降维的过程。
"""
# 示例：
# 这个例子说明了一个学习的距离度量，最大限度地提高了最近邻分类的准确性。与原始点空间相比，它提供了此度量的直观表示。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from matplotlib import cm
from scipy.special import logsumexp

X, y = make_classification(n_samples=9, n_features=2, n_informative=2,
                           n_redundant=0, n_classes=3, n_clusters_per_class=1,
                           class_sep=1.0, random_state=0)

plt.figure(1)
ax = plt.gca()
for i in range(X.shape[0]):  # 9次迭代  绘制样本点
    ax.text(X[i, 0], X[i, 1], str(i), va='center', ha='center')
    ax.scatter(X[i, 0], X[i, 1], s=300, c=cm.Set1(y[[i]]), alpha=0.4)

ax.set_title("Original points")
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.axis('equal')  # so that boundaries are displayed correctly as circles


def link_thickness_i(X, i):
    diff_embedded = X[i] - X  # 每个点到样本点3的差值 9*2
    dist_embedded = np.einsum('ij,ij->i', diff_embedded, diff_embedded)  # 计算距离
    # einsum 爱因斯坦求和
    dist_embedded[i] = np.inf

    # 计算指数距离 (使用log-sum-exp 避免数值不稳定
    exp_dist_embedded = np.exp(-dist_embedded - logsumexp(-dist_embedded))
    return exp_dist_embedded


def relate_point(X, i, ax):
    pt_i = X[i]
    for j, pt_j in enumerate(X):
        thickness = link_thickness_i(X, i)  # 计算每个点与点3之间的距离。
        if i != j:
            line = ([pt_i[0], pt_j[0]], [pt_i[1], pt_j[1]])
            ax.plot(*line, c=cm.Set1(y[j]), linewidth=5*thickness[j])  # 绘制连接线，距离越近，线越粗
            # ax.plot(*line, c=cm.Set1(y[j]))


i = 3
relate_point(X, i, ax)

# #############################################
# 使用 NeighborhoodComponentsAnalysis转换
nca = NeighborhoodComponentsAnalysis(max_iter=30, random_state=0)
nca = nca.fit(X, y)

plt.figure(2)
ax2 = plt.gca()
X_embedded = nca.transform(X)
relate_point(X_embedded, i, ax2)

for i in range(len(X)):
    ax2.text(X_embedded[i, 0], X_embedded[i, 1], str(i), va='center', ha='center')
    ax2.scatter(X_embedded[i, 0], X_embedded[i, 1], s=300, c=cm.Set1(y[[i]]), alpha=0.4)

ax2.set_title("NCA embedding")
ax2.axes.get_xaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)
ax2.axis('equal')

plt.show()

