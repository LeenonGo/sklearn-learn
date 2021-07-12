# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/12 14:51
# @Function: 最近邻回归  https://www.scikitlearn.com.cn/0.21.3/7/#163
"""
预测值由最近邻标签的均值计算而来的
KNeighborsRegressor 基于每个查询点的 k 个最近邻实现， 其中 k 是用户指定的整数值。
RadiusNeighborsRegressor 基于每个查询点的固定半径 r 内的邻点数量实现， 其中 r 是用户指定的浮点数值。

权重的使用与分类方法中一致
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

y[::5] += 1 * (0.5 - np.random.rand(8))

n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(T)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, color='darkorange', label='data')
    plt.plot(T, y_, color='navy', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))

plt.tight_layout()
plt.show()
