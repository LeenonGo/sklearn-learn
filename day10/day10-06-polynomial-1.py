# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/5 14:13
# @Function: 多项式回归 https://www.scikitlearn.com.cn/0.21.3/2/#1116
# 用基函数展开线性模型

from sklearn.preprocessing import PolynomialFeatures
import numpy as np
X = np.arange(6).reshape(3, 2)
print(X)
poly = PolynomialFeatures(degree=2)  # X 的特征已经从 [x1, x2] 转换到 [1, x1, x2, x1^2, x1 x2, x2^2]
print(poly.fit_transform(X))
