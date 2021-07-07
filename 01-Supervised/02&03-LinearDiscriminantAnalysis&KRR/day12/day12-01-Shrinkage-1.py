# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/7 10:28
# @Function: 收缩  https://www.scikitlearn.com.cn/0.21.3/3/#124-shrinkage
# 收缩是一种在训练样本数量相比特征而言很小的情况下可以提升的协方差矩阵预测（准确性）的工具。
# 通过线性判别分析的参数shrinkage 实现
# 参数设置0-1。0表示无收缩（使用经验协方差）；1表示完全收缩（协方差的对角矩阵将被当做协方差的估计）
#
# 示例：协方差估计法和Oracle收缩近似如何改进分类
#
#
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import OAS


n_train = 20  # samples for training
n_test = 200  # samples for testing
n_averages = 50  # how often to repeat classification
n_features_max = 75  # maximum number of features
step = 4  # step size for the calculation


def generate_data(n_samples, n_features):
    """生成具有噪声特征的随机斑点数据.

    该方法返回一个(n_samples, n_features)的数组，一个n_samples的目标标签

    只有一个特征包含鉴别信息，其他特征只包含噪声。
    """
    X, y = make_blobs(n_samples=n_samples, n_features=1, centers=[[-2], [2]])

    # 添加无判别性的特征
    if n_features > 1:
        X = np.hstack([X, np.random.randn(n_samples, n_features - 1)])
    return X, y


acc_clf1, acc_clf2, acc_clf3 = [], [], []
n_features_range = range(1, n_features_max + 1, step)
for n_features in n_features_range:
    score_clf1, score_clf2, score_clf3 = 0, 0, 0
    for _ in range(n_averages):
        X, y = generate_data(n_train, n_features)

        oa = OAS(store_precision=False, assume_centered=False)
        clf1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, y)
        clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X, y)  # 无收缩
        clf3 = LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=oa).fit(X, y)

        X, y = generate_data(n_test, n_features)
        score_clf1 += clf1.score(X, y)
        score_clf2 += clf2.score(X, y)
        score_clf3 += clf3.score(X, y)

    acc_clf1.append(score_clf1 / n_averages)
    acc_clf2.append(score_clf2 / n_averages)
    acc_clf3.append(score_clf3 / n_averages)

features_samples_ratio = np.array(n_features_range) / n_train

plt.plot(features_samples_ratio, acc_clf1, linewidth=2, label="Linear Discriminant Analysis with Ledoit Wolf", color='navy')
plt.plot(features_samples_ratio, acc_clf2, linewidth=2, label="Linear Discriminant Analysis", color='gold')
plt.plot(features_samples_ratio, acc_clf3, linewidth=2, label="Linear Discriminant Analysis with OAS", color='red')

plt.xlabel('n_features / n_samples')
plt.ylabel('Classification accuracy')

plt.legend(loc=3, prop={'size': 12})
plt.suptitle('Linear Discriminant Analysis vs. ' + '\n'
             + 'Shrinkage Linear Discriminant Analysis vs. ' + '\n'
             + 'OAS Linear Discriminant Analysis (1 discriminative feature)')
plt.show()
