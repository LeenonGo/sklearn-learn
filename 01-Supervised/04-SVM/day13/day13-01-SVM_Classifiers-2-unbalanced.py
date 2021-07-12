# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/9 11:23
# @Function: 非均衡问题  https://www.scikitlearn.com.cn/0.21.3/5/#1413
# 该问题是给某一个类或个别样本提高权重（class_weight 和 sample_weight）

"""
对于不平衡的类，使用SVC寻找最优的分离超平面。

首先用一个简单的SVC找到分离平面，然后绘制（虚线）分离超平面，并对不平衡类进行自动校正。
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

n_samples_1 = 1000
n_samples_2 = 100
centers = [[0.0, 0.0], [2.0, 2.0]]
clusters_std = [1.5, 0.5]
X, y = make_blobs(n_samples=[n_samples_1, n_samples_2], centers=centers,
                  cluster_std=clusters_std, random_state=0, shuffle=False)

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# class_weight：{class_label:value},value是浮点数大0的值,把类class_label的参数C设置为C*value.
wclf = svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# 获取分离超平面
Z = clf.decision_function(xy).reshape(XX.shape)
# 绘制决策边界
a = ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

# get the separating hyperplane for weighted classes
Z = wclf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins for weighted classes
b = ax.contour(XX, YY, Z, colors='r', levels=[0], alpha=0.5, linestyles=['-'])

plt.legend([a.collections[0], b.collections[0]], ["non weighted", "weighted"],
           loc="upper right")
plt.show()

