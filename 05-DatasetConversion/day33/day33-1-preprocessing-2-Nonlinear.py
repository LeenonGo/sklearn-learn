# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/16 16:27
# @Function: 5.3.2 非线性转换
#

# 有两种类型的转换是可用的:分位数转换和幂函数转换。
# 分位数和幂变换都基于特征的单调变换，从而保持了每个特征值的秩。


# 1 映射到均匀分布
# QuantileTransformer 类以及 quantile_transform 函数提供了一个基于分位数函数的无参数转换，将数据映射到了零到一的均匀分布上:
import numpy as np

from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)
print(np.percentile(X_train[:, 0], [0, 25, 50, 75, 100]))
# 这个特征是萼片的厘米单位的长度。一旦应用分位数转换，这些元素就接近于之前定义的百分位数:
print(np.percentile(X_train_trans[:, 0], [0, 25, 50, 75, 100]))


# 2 映射到高斯分布
# 在许多建模场景中，需要数据集中的特征的正态化。
# 幂变换是一类参数化的单调变换， 其目的是将数据从任何分布映射到尽可能接近高斯分布，以便稳定方差和最小化偏斜。
# 类 PowerTransformer 目前提供两个这样的幂变换,Yeo-Johnson transform 和 the Box-Cox transform。













