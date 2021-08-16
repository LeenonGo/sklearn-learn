# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/16 17:02
# @Function: 5.4 缺失值插补 https://www.scikitlearn.com.cn/0.21.3/41/
# 
# 因为各种各样的原因，真实世界中的许多数据集都包含缺失数据，这类数据经常被编码成空格、NaNs，或者是其他的占位符。
# 但是这样的数据集并不能scikit-learn学习算法兼容

# 1 单变量与多变量插补
# 一种类型的插补算法是单变量算法，它只使用第i个特征维度中的非缺失值(如impute.SimpleImputer)来插补第i个特征维中的值。
# 相比之下，多变量插补算法使用整个可用特征维度来估计缺失的值(如impute.IterativeImputer)。


# 2 单变量插补
# SimpleImputer类提供了计算缺失值的基本策略。
# 缺失值可以用提供的常数值计算，也可以使用缺失值所在的行/列中的统计数据(平均值、中位数或者众数)来计算。
# 这个类也支持不同的缺失值编码。

# 以下代码段演示了如何使用包含缺失值的列(轴0)的平均值来替换编码为 np.nan 的缺失值:
import numpy as np
# from sklearn.preprocessing import Imputer
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imp.fit([[1, 2], [np.nan, 3], [7, 6]])
# X = [[np.nan, 2], [6, np.nan], [7, 6]]
# print(imp.transform(X))
#
# SimpleImputer类也支持稀疏矩阵


# 3.1 多变量插补的灵活性
# 3.2 单次与多次插补

# 5 标记缺失值
# MissingIndicator转换器用于将数据集转换为相应的二进制矩阵，以指示数据集中缺失值的存在。
# 这个变换与归算结合起来是有用的。当使用插补时，保存关于哪些值丢失的信息可以提供有用的信息。










