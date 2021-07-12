# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/9 10:49
# @Function: SVM分类
#
#


"""
示例：为四个不同核的SVM分类器绘制决策面。
LinearSVC() and SVC(kernel='linear') 产生略微不同的决策边界：原因
    1. LinearSVC最小化平方hinge损失。SVC最小化规则hinge损失。
    2. LinearSVC使用OVR降维。SVC使用OVO降维。

通常情况下OVR分类使用更多，因为二者分类结果基本相似，但时间显著减少

两种线性模型都具有线性决策边界（相交超平面），
而非线性核模型（多项式或高斯RBF）具有更灵活的非线性决策边界，其形状取决于核的种类及其参数。

关于OVO和OVR可参见\00-before\day02\day02-02-2.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def make_meshgrid(x, y, h=.02):
    """
    创建一个网格
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """
    绘制等高线
    为分类绘制决策边界
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# 创建一个SVM实例并拟合数据。由于要绘制支持向量，不缩放数据
C = 1.0  # SVM正则化参数
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C)
          )
models = (clf.fit(X, y) for clf in models)

titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
