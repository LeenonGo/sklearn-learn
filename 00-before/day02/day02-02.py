"""
使用核   https://www.scikitlearn.com.cn/0.21.3/54/#_8

在特征空间中 类并不总是线性可分的。当不是线性可分时，解决办法就是构建一个不是线性的但能是多项式的函数做代替。
核技巧(kernel trick)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 显示3种不同类型的核函数 多项式和径向基函数在数据点不可线性分离时特别有用。

# Our dataset and targets
X = np.c_[
    (.4, -.7), (-1.5, -1), (-1.4, -.9), (-1.3, -1.2), (-1.1, -.2), (-1.2, -.4), (-.5, 1.2), (-1.5, 2.1),
    (1, 1), (1.3, .8), (1.2, .5), (.2, -2), (.5, -2.4), (.2, -2.3), (0, -2.7), (1.3, 2.1)
    ].T
Y = [0] * 8 + [1] * 8
# Y = [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4  # 散点图绘制时， c=Y, 颜色描绘

# figure number
fignum = 1

# fit the model
for kernel in ('rbf',):  # 线性 多项式 径向基函数
    # for kernel in ('linear', 'poly', 'rbf'):  # 线性 多项式 径向基函数
    clf = svm.SVC(kernel=kernel, gamma=2)
    clf.fit(X, Y)

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(8, 6))
    plt.clf()
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    # print(clf.support_vectors_[:, 0])
    # print(clf.support_vectors_)  # 获得支持向量  离最优分类平面最近的离散点
    # print("------------------------------")
    # print(clf.n_support_)  # 获得每个类支持向量的个数
    # print("------------------------------")
    # print(clf.support_)  # 获得支持向量点在原数据中的下标
    # 关于支持向量的介绍，查看博客https://blog.csdn.net/Tong_T/article/details/78918068
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired, edgecolors='k')

    plt.axis('tight')
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    ZZ = np.c_[XX.ravel(), YY.ravel()]
    Z = clf.decision_function(ZZ)  # 在day02-02-2.py中学习decision_function函数
    # 这里得到的是对坐标的预测值 方法同day01-05-2.py

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])  # 轮廓
    #
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1
plt.show()
