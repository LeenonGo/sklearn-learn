# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/1 12:59
# @Function: 贝叶斯岭回归小例子
# 

from sklearn import linear_model

X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
Y = [0., 1., 2., 3.]
reg = linear_model.BayesianRidge()
reg.fit(X, Y)  # 训练

print(reg.predict([[1, 0.]]))  # 预测
print(reg.coef_)  # 输出权值
