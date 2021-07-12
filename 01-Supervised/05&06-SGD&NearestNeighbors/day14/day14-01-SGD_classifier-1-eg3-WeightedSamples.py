# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/12 13:00
# @Function: Weighted samples
# 绘制加权数据集的决策函数，其中点的大小与其权重成正比。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

np.random.seed(0)
X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]  # 20*2
y = [1] * 10 + [-1] * 10
sample_weight = 100 * np.abs(np.random.randn(20))
# 对最后10个样本赋予更大的权重
sample_weight[:10] *= 10

xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, s=sample_weight, alpha=0.9, cmap=plt.cm.bone, edgecolor='black')
# plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.9, cmap=plt.cm.bone, edgecolor='black')

clf = SGDClassifier(alpha=0.01, max_iter=100)
clf.fit(X, y)
print(clf.coef_)
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
no_weights = plt.contour(xx, yy, Z, levels=[0], linestyles=['solid'])

# fit the weighted model
clf = SGDClassifier(alpha=0.01, max_iter=100)
clf.fit(X, y, sample_weight=sample_weight)
print(clf.coef_)
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
samples_weights = plt.contour(xx, yy, Z, levels=[0], linestyles=['dashed'])

plt.legend([no_weights.collections[0], samples_weights.collections[0]],
           ["no weights", "with weights"], loc="lower left")

plt.xticks(())
plt.yticks(())
plt.show()


"""
图中实线为不考虑数据权重的情况下对样本点的分类
虚线为考虑数据权重的情况下对样本的分类
可以看出
"""
