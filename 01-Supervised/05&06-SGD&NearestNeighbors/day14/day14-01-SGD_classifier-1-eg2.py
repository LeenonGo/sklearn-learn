# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/12 11:23
# @Function: 示例：基于 iris（鸢尾花）数据集上的 OVA 方法
# 虚线表示三个 OVA 分类器，不同背景色代表由三个分类器产生的决策面。

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import SGDClassifier

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

# 打乱顺序
idx = np.arange(X.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# 标准化数据
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

clf = SGDClassifier(alpha=0.001, max_iter=100).fit(X, y)

h = .02

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 绘制决策边界.
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis('tight')

# # 绘制训练数据
colors = "bry"
for i, color in zip(clf.classes_, colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i], cmap=plt.cm.Paired, edgecolor='black', s=20)
plt.title("Decision surface of multi-class SGD")
plt.axis('tight')
#
# 绘制三个one-against-all分类器
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
coef = clf.coef_  # 3*2的矩阵，三个类别，2个特征的权重
intercept = clf.intercept_


def plot_hyperplane(c, color):
    def line(x0):
        return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
        # 返回的是坐标中X对应的Y值。因为在式子中，斜率（即权重coef)和截距（intercept）是已知的。
    plt.plot([xmin, xmax], [line(xmin), line(xmax)], ls="--", color=color)


for i, color in zip(clf.classes_, colors):
    plot_hyperplane(i, color)
plt.legend()
plt.show()


