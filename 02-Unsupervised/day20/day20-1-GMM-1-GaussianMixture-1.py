# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/26 13:14
# @Function: 高斯混合模型
#
# sklearn.mixture 是一个应用高斯混合模型进行非监督学习的包(支持 diagonal，spherical，tied，full 四种协方差矩阵),
# 它可以对数据进行抽样，并且根据数据来估计模型。同时该包也支持由用户来决定模型内混合的分量数量
# 在高斯混合模型中，我们将每一个高斯分布称为一个分量，即 component
#       diagonal  指每个分量有各自独立的对角协方差矩阵，
#       spherical 指每个分量有各自独立的方差(再注:spherical是一种特殊的 diagonal, 对角的元素相等)，
#       tied      指所有分量共享一个标准协方差矩阵，
#       full      指每个分量有各自独立的标准协方差矩阵，
# 高斯混合模型是一个假设所有的数据点都是生成于有限个带有未知参数的高斯分布所混合的概率模型。


# 示例：高斯混合模型的密度估计
# 绘制两个高斯混合的密度估计。数据由两个具有不同中心和协方差矩阵的高斯函数生成。

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

n_samples = 300

# generate random sample, two components
np.random.seed(0)

# 生成以（20，20）为中心的球形数据
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

# 生成零中心拉伸高斯数据
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

# 将两个数据集连接到最终的训练集中
X_train = np.vstack([shifted_gaussian, stretched_gaussian])

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(X_train)

# 以等高线图的形式显示模型预测的分数
x = np.linspace(-20., 30.)
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()


