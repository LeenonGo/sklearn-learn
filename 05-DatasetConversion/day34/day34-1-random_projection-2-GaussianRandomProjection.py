# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/17 12:30
# @Function: 5.6.2. 高斯随机投影
#  sklearn.random_projection.GaussianRandomProjection 通过将原始输入空间投影到随机生成的矩阵降低维度
#  （该矩阵的组件由该分布中抽取 N(0, 1/n_components)
#

import numpy as np
from sklearn import random_projection
X = np.random.rand(100, 10000)
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.shape)
