# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/6 12:58
# @Function: 分解成分中的信号（矩阵分解问题） https://www.scikitlearn.com.cn/0.21.3/24/
# 2.5.1. 主成分分析（PCA） https://www.scikitlearn.com.cn/0.21.3/24/#251-pca

# PCA 用于对具有一组连续正交分量的多变量数据集进行方差最大化的分解
# 在应用SVD(奇异值分解) 之前, PCA 是在为每个特征聚集而不是缩放输入数据。
# 可选参数 whiten=True 使得可以将数据投影到奇异（singular）空间上，同时将每个成分缩放到单位方差。

# 示例1 见day11-01-LDA-1.py  该数据集包含4个特征，通过PCA降维后投影到方差最大的二维空间上


# PCA 对象还提供了 PCA 算法的概率解释，其可以基于可解释的方差量给出数据的可能性。
# PCA对象实现了在交叉验证（cross-validation）中使用 score 方法：

# 概率主成分分析和因子分析是概率模型。其结果是，新数据的可能性可用于模型选择和协方差估计。
# 在这里，我们将PCA和FA与交叉验证相比较，这些交叉验证是针对同方差噪声（每个特征的噪声方差相同）或异方差噪声（每个特征的噪声方差不同）损坏的低秩数据。
# 在第二步中，我们将模型似然与从收缩协方差估计中获得的似然进行比较。

# 我们可以观察到，对于同方差噪声，FA和PCA都成功地恢复了低秩子空间的大小。在这种情况下，PCA的可能性高于FA。
# 然而，当存在异方差噪声时，主成分分析失败并高估了秩。在适当的情况下，低阶模型比收缩模型更可能出现。

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# #############################################################################
rng = np.random.RandomState(42)
# Create the data

n_samples, n_features, rank = 1000, 50, 10
sigma = 1.
U, _, _ = linalg.svd(rng.randn(n_features, n_features))
X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)

# Adding homoscedastic noise  同方差
X_homo = X + sigma * rng.randn(n_samples, n_features)

# Adding heteroscedastic noise  异方差
sigmas = sigma * rng.rand(n_features) + sigma / 2.
X_hetero = X + rng.randn(n_samples, n_features) * sigmas

# #############################################################################
# Fit the models

n_components = np.arange(0, n_features, 5)  # options for n_components


def compute_scores(X):
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()

    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))

    return pca_scores, fa_scores


def shrunk_cov_score(X):
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))


def lw_score(X):
    return np.mean(cross_val_score(LedoitWolf(), X))


for X, title in [(X_homo, 'Homoscedastic Noise'), (X_hetero, 'Heteroscedastic Noise')]:
    pca_scores, fa_scores = compute_scores(X)
    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]

    pca = PCA(svd_solver='full', n_components='mle')
    pca.fit(X)
    n_components_pca_mle = pca.n_components_

    print("best n_components by PCA CV = %d" % n_components_pca)
    print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
    print("best n_components by PCA MLE = %d" % n_components_pca_mle)

    plt.figure()
    plt.plot(n_components, pca_scores, 'b', label='PCA scores')
    plt.plot(n_components, fa_scores, 'r', label='FA scores')
    plt.axvline(rank, color='g', label='TRUTH: %d' % rank, linestyle='-')
    plt.axvline(n_components_pca, color='b', label='PCA CV: %d' % n_components_pca, linestyle='--')
    plt.axvline(n_components_fa, color='r', label='FactorAnalysis CV: %d' % n_components_fa, linestyle='--')
    plt.axvline(n_components_pca_mle, color='k', label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

    # compare with other covariance estimators
    plt.axhline(shrunk_cov_score(X), color='violet', label='Shrunk Covariance MLE', linestyle='-.')
    plt.axhline(lw_score(X), color='orange', label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')

    plt.xlabel('nb of components')
    plt.ylabel('CV scores')
    plt.legend(loc='lower right')
    plt.title(title)

plt.show()









