# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/13 17:05
# @Function: 5.1.2. 回归中的目标转换
# TransformedTargetRegressor在拟合回归模型之前对目标y进行转换。这些预测通过一个逆变换被映射回原始空间。
# 它以预测所用的回归器为参数，将应用于目标变量的变压器为参数:


import numpy as np
from sklearn.datasets import load_boston
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
boston = load_boston()
X = boston.data
y = boston.target
transformer = QuantileTransformer(output_distribution='normal')
regressor = LinearRegression()
regr = TransformedTargetRegressor(regressor=regressor, transformer=transformer)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
regr.fit(X_train, y_train)
print('R2 score: {0:.2f}'.format(regr.score(X_test, y_test)))
raw_target_regr = LinearRegression().fit(X_train, y_train)
print('R2 score: {0:.2f}'.format(raw_target_regr.score(X_test, y_test)))
# 默认情况下，所提供的函数在每次匹配时都被检查为彼此的倒数。但是，可以通过将check_reverse设置为False来绕过这个检查

