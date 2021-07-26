# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/26 15:37
# @Function:
# 示例2： 高斯混合模型椭球

# 下面的例子将具有固定数量分量的高斯混合模型与具有Dirichlet process prior（狄利克雷过程先验）的变分高斯混合模型进行了比较。
# 此处，一个经典高斯混合模型被指定由 5 个分量（其中两个是聚类）在某个数据集上拟合而成。
# 我们可以看到，具有狄利克雷过程的变分高斯混合模型 在相同的数据上进行拟合时,可以将自身分量数量制在 2 。
# 在该例子中，用户选择了 n_components=5 ，这与真正的试用数据集（toy dataset）的生成分量数量不符。
# 很容易注意到, 狄利克雷过程先验的变分高斯混合模型可以采取保守的策略，仅拟合生成一个分量。

# 用期望最大化（GaussianMixture 类）和变分推理（具有Dirichlet过程先验的bayesiangaussian混合类模型）绘制两个gaussian混合模型的置信椭球。
# 这两个模型都可以使用五个分量来拟合数据。请注意，期望最大化模型将必然使用所有五个分量，而变分推理模型将有效地仅使用良好拟合所需的数量。
# 我们可以看到，期望最大化模型由于试图拟合过多的分量而任意拆分了一些成分，而Dirichlet过程模型则自动调整状态数。
# 这个例子没有说明这一点，我们在低维空间中，
# 但Dirichlet过程模型的另一个优点是，由于推理算法的正则化特性，它可以有效地拟合完全协方差矩阵，即使每个簇的示例数少于数据中的维数。

import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])


def plot_results(X, Y_, means, covariances, index, title):
    """
    :param X:
    :param Y_: 预测值
    :param means: 均值
    :param covariances: 协方差
    :param index:
    :param title:
    :return:
    """
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


n_samples = 500

np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C), .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

# Fit a Gaussian mixture with EM using five components
gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0, 'Gaussian Mixture')

# Fit a Dirichlet process Gaussian mixture using five components
dpgmm = mixture.BayesianGaussianMixture(n_components=5, covariance_type='full').fit(X)
plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
             'Bayesian Gaussian Mixture with a Dirichlet process prior')

plt.show()

