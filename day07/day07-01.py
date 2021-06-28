# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/6/28 9:02
# @Function:  普通最小二乘法LinearRegression  https://www.scikitlearn.com.cn/0.21.3/2/#111
# 使用糖尿病数据集的第一个特征，画出一条直线，该直线将使真实值和预测值之间的残差平方和最小化。

# 残差平方和(RSS): 等同于误差项平方和。差的平方之和

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]  #
# diabetes_X[:, 2] 取出的是数据集第三个特征值，是一组array，使用np.newaxis使其转换成矩阵

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

regr = linear_model.LinearRegression()

regr.fit(diabetes_X_train, diabetes_y_train)  # 训练模型
diabetes_y_pred = regr.predict(diabetes_X_test)  # 预测

print('Coefficients: \n', regr.coef_)
print('intercept_: \n', regr.intercept_)
print('Mean squared error: %.2f' % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print('Coefficient of determination: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))  # 确定参数， 该参数越大越完美

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

plt.show()
