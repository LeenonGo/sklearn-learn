# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/8 13:54
# @Function: Novelty Detection（新奇点检测）
# 考虑一个以 p 个特征描述、包含 n 个有着相同分布的观测值的数据集。
# 现在考虑我们再往该数据集中添加一个观测值。
# 如果新观测与原有观测有很大差异，我们就可以怀疑它是否是内围值(regular吗？ （即是否来自同一分布？）
# 如果新观测与原有观测很相似，我们就无法将其与原有观测区分开吗？ 这就是新奇点检测工具和方法所解决的问题。

# 要学习出一个粗略且紧密的边界，界定出初始观测分布的轮廓，绘制在嵌入的 p 维空间中。
# 那么，如果后续的观测值都落在这个边界划分的子空间内，则它们被认为来自与初始观测值相同的总体。
# 否则，如果它们在边界之外，我们可以说就我们评估中给定的置信度而言，它们是异常的。

# One-Class SVM（单类支持向量机）已经采用以实现新奇检测，并在 支持向量机 模块的 svm.OneClassSVM 对象中实现。

# 示例：通过 svm.OneClassSVM 对象学习一些数据来将边界可视化。
# OneClassSVM是一种无监督的算法，它学习一个用于新奇点检测的决策函数：将新数据分类为与训练集相似或不同的数据。

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# Generate train data
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s, edgecolors='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s, edgecolors='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations", "new regular observations", "new abnormal observations"],
           loc="upper left", prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "error train: %d/200 ; errors novel regular: %d/40 ; "
    "errors novel abnormal: %d/40"
    % (n_error_train, n_error_test, n_error_outliers))
plt.show()

