# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/17 12:51
# @Function: 5.8. 成对的矩阵, 类别和核函数
#  sklearn.metrics.pairwise 子模块实现了用于评估成对距离或样本集合之间的联系的实用程序。

# 核函数是相似度的标准. 如果对象 a 和 b 被认为 “更加相似” 相比对象 a 和 c，那么 s(a, b) > s(a, c). 核函数必须是半正定性的.
# 存在许多种方法将距离度量转换为相似度标准，例如核函数。 假定 D 是距离, S 是核函数:
#   1. S = np.exp(-D * gamma), 其中 gamma 的一种选择是 1 / num_features
#   2. S = 1. / (D / np.max(D))
#

# X 的行向量和 Y 的行向量之间的距离可以用函数 pairwise_distances 进行计算。

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
X = np.array([[2, 3], [3, 5], [5, 8]])
Y = np.array([[1, 0], [2, 1]])
print(pairwise_distances(X, Y, metric='manhattan'))
# 如果 Y 被忽略，则 X 的所有行向量的成对距离就会被计算。
print(pairwise_distances(X, metric='manhattan'))
print(pairwise_kernels(X, Y, metric='linear'))

