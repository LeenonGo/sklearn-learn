# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/16 16:51
# @Function: 5.3.7 生成多项式特征
# 在机器学习中，通过增加一些输入数据的非线性特征来增加模型的复杂度通常是有效的。
# 一个简单通用的办法是使用多项式特征，这可以获得特征的更高维度和互相间关系的项。这在 PolynomialFeatures 中实现
#
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3, 2)
print(X)

poly = PolynomialFeatures(2)
print(poly.fit_transform(X))
# X 的特征已经从 (X_1, X_2) 转换为 (1, X_1, X_2, X_1^2, X_1X_2, X_2^2) 。

# 在一些情况下，只需要特征间的交互项，这可以通过设置 interaction_only=True 来得到
X = np.arange(9).reshape(3, 3)
print(X)
poly = PolynomialFeatures(degree=3, interaction_only=True)
print(poly.fit_transform(X))
# X的特征已经从 (X_1, X_2, X_3) 转换为 (1, X_1, X_2, X_3, X_1X_2, X_1X_3, X_2X_3, X_1X_2X_3) 。


