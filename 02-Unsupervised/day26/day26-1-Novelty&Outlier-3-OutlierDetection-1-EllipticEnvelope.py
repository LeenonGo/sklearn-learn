# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/8 15:01
# @Function: 2.7.3.1. Fitting an elliptic envelope（椭圆模型拟合）
# https://www.scikitlearn.com.cn/0.21.3/26/#2731-fitting-an-elliptic-envelope
# 

# 实现离群点检测的一种常见方式是假设常规数据来自已知分布
# 试图定义数据的 “形状”，并且可以将偏远观测(outlying observation)定义为足够远离拟合形状的观测。

# 示例：说明对位置和协方差使用标准估计 (covariance.EmpiricalCovariance) 或稳健估计 (covariance.MinCovDet) 来评估观测值的偏远性的差异。
# 这个例子显示了高斯分布数据上马氏距离的协方差估计。

# 最好使用稳健的协方差估计器，以确保该估计器能够抵抗数据集中的“错误”观测，并且计算出的马氏距离能够准确反映观测的真实组织。

# 此示例说明了马氏距离是如何受外围数据影响的。当使用基于标准协方差MLE的马氏距离时，来自污染分布的观测值与来自真实高斯分布的观测值无法区分。
# 使用基于MCD的马氏距离，这两个种群变得可区分。相关应用包括离群点检测、观测排序和聚类。

# 首先，生成一个包含125个样本和2个特征的数据集。
# 两个特征均为高斯分布，平均值为0，但特征1的标准偏差为2，特征2的标准偏差为1。
#
# 接下来，将25个样本替换为高斯异常样本，其中特征1的标准偏差为1，特征2的标准偏差为7。

import numpy as np

np.random.seed(7)

n_samples = 125
n_outliers = 25
n_features = 2

# generate Gaussian data of shape (125, 2)
gen_cov = np.eye(n_features)  # 生成对角矩阵
gen_cov[0, 0] = 2.
X = np.dot(np.random.randn(n_samples, n_features), gen_cov)

outliers_cov = np.eye(n_features)  # add some outliers
outliers_cov[np.arange(1, n_features), np.arange(1, n_features)] = 7.
X[-n_outliers:] = np.dot(np.random.randn(n_outliers, n_features), outliers_cov)

# 下面，我们将基于MCD和MLE的协方差估计值拟合到我们的数据中，并打印估计的协方差矩阵。
# 请注意，使用基于MLE的估计器（7.5）时，特征2的估计方差远高于MCD稳健估计器（1.2）。
# 这表明基于MCD的稳健估计对异常样本的抵抗力更强，这些样本的设计目的是在特征2中具有更大的方差。
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance, MinCovDet

# fit a MCD robust estimator to data
robust_cov = MinCovDet().fit(X)
# fit a MLE estimator to data
emp_cov = EmpiricalCovariance().fit(X)
print('Estimated covariance matrix:\n'
      'MCD (Robust):\n{}\n'
      'MLE:\n{}'.format(robust_cov.covariance_, emp_cov.covariance_))

# 为了更好地显示差异，我们绘制了用这两种方法计算的马氏距离的等高线。
# 请注意，基于MCD的稳健马氏距离更适合内部黑点，而基于MLE的距离更受异常红点的影响。
fig, ax = plt.subplots(figsize=(10, 5))
# Plot data set
inlier_plot = ax.scatter(X[:, 0], X[:, 1], color='black', label='inliers')
outlier_plot = ax.scatter(X[:, 0][-n_outliers:], X[:, 1][-n_outliers:], color='red', label='outliers')
ax.set_xlim(ax.get_xlim()[0], 10.)
ax.set_title("Mahalanobis distances of a contaminated data set")

# Create meshgrid of feature 1 and feature 2 values
xx, yy = np.meshgrid(np.linspace(plt.xlim()[0], plt.xlim()[1], 100), np.linspace(plt.ylim()[0], plt.ylim()[1], 100))
zz = np.c_[xx.ravel(), yy.ravel()]
# Calculate the MLE based Mahalanobis distances of the meshgrid
mahal_emp_cov = emp_cov.mahalanobis(zz)
mahal_emp_cov = mahal_emp_cov.reshape(xx.shape)
emp_cov_contour = plt.contour(xx, yy, np.sqrt(mahal_emp_cov), cmap=plt.cm.PuBu_r, linestyles='dashed')

# Calculate the MCD based Mahalanobis distances
mahal_robust_cov = robust_cov.mahalanobis(zz)
mahal_robust_cov = mahal_robust_cov.reshape(xx.shape)
robust_contour = ax.contour(xx, yy, np.sqrt(mahal_robust_cov), cmap=plt.cm.YlOrBr_r, linestyles='dotted')

# Add legend
ax.legend([emp_cov_contour.collections[1], robust_contour.collections[1],
          inlier_plot, outlier_plot],
          ['MLE dist', 'MCD dist', 'inliers', 'outliers'],
          loc="upper right", borderaxespad=0)


# 最后，我们强调了基于MCD的马氏距离区分异常值的能力。
# 我们取马氏距离的立方根，得到近似正态分布，然后用箱线图绘制内点和离群点样本的值。
# 对于稳健的基于MCD的马氏距离，离群样本的分布与内部样本的分布更为分离。
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.subplots_adjust(wspace=.6)

# Calculate cubic root of MLE Mahalanobis distances for samples
emp_mahal = emp_cov.mahalanobis(X - np.mean(X, 0)) ** (0.33)
# Plot boxplots
ax1.boxplot([emp_mahal[:-n_outliers], emp_mahal[-n_outliers:]], widths=.25)
# Plot individual samples
ax1.plot(np.full(n_samples - n_outliers, 1.26), emp_mahal[:-n_outliers], '+k', markeredgewidth=1)
ax1.plot(np.full(n_outliers, 2.26), emp_mahal[-n_outliers:], '+k', markeredgewidth=1)
ax1.axes.set_xticklabels(('inliers', 'outliers'), size=15)
ax1.set_ylabel(r"$\sqrt[3]{\rm{(Mahal. dist.)}}$", size=16)
ax1.set_title("Using non-robust estimates\n(Maximum Likelihood)")

# Calculate cubic root of MCD Mahalanobis distances for samples
robust_mahal = robust_cov.mahalanobis(X - robust_cov.location_) ** (0.33)
# Plot boxplots
ax2.boxplot([robust_mahal[:-n_outliers], robust_mahal[-n_outliers:]], widths=.25)
# Plot individual samples
ax2.plot(np.full(n_samples - n_outliers, 1.26), robust_mahal[:-n_outliers], '+k', markeredgewidth=1)
ax2.plot(np.full(n_outliers, 2.26), robust_mahal[-n_outliers:], '+k', markeredgewidth=1)
ax2.axes.set_xticklabels(('inliers', 'outliers'), size=15)
ax2.set_ylabel(r"$\sqrt[3]{\rm{(Mahal. dist.)}}$", size=16)
ax2.set_title("Using robust estimates\n(Minimum Covariance Determinant)")

plt.show()
