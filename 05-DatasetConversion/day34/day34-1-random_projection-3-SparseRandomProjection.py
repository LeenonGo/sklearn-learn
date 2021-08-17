# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/17 12:35
# @Function: 5.6.3. 稀疏随机矩阵
# sklearn.random_projection.SparseRandomProjection 使用稀疏随机矩阵，通过投影原始输入空间来降低维度。
# 稀疏矩阵可以替换高斯随机投影矩阵来保证相似的嵌入质量，且内存利用率更高、投影数据的计算更快。

import numpy as np
from sklearn import random_projection
X = np.random.rand(100,10000)
transformer = random_projection.SparseRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.shape)

