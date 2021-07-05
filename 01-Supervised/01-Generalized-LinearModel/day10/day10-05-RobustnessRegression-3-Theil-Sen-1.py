# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/5 13:46
# @Function: Theil-Sen 预估器: 广义中值估计器（generalized-median-based estimator）
# 使用中位数在多个维度泛化，对多元异常值更具有鲁棒性，
# 但问题是，随着维数的增加，估计器的准确性在迅速下降。准确性的丢失，导致在高维上的估计值比不上普通的最小二乘法。
#
# 示例：稳健线性估计拟合
#

import numpy as np
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import (LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)

np.random.seed(42)

X = np.random.normal(size=400)
y = np.sin(X)
X = X[:, np.newaxis]  # 数据二维化

X_test = np.random.normal(size=200)
y_test = np.sin(X_test)
X_test = X_test[:, np.newaxis]

# ---------处理异常数据---------
y_errors = y.copy()
y_errors[::3] = 3

X_errors = X.copy()
X_errors[::3] = 3

y_errors_large = y.copy()
y_errors_large[::3] = 10

X_errors_large = X.copy()
X_errors_large[::3] = 10
# -----------------------------


estimators = [('OLS', LinearRegression()),
              ('Theil-Sen', TheilSenRegressor(random_state=42)),
              ('RANSAC', RANSACRegressor(random_state=42)),
              ('HuberRegressor', HuberRegressor())]
colors = {'OLS': 'turquoise', 'Theil-Sen': 'gold', 'RANSAC': 'lightgreen', 'HuberRegressor': 'black'}
linestyle = {'OLS': '-', 'Theil-Sen': '-.', 'RANSAC': '--', 'HuberRegressor': '--'}
lw = 3

x_plot = np.linspace(X.min(), X.max())
for title, this_X, this_y in [
        ('Modeling Errors Only', X, y),
        ('Corrupt X, Small Deviants', X_errors, y),
        ('Corrupt y, Small Deviants', X, y_errors),
        ('Corrupt X, Large Deviants', X_errors_large, y),
        ('Corrupt y, Large Deviants', X, y_errors_large)]:
    plt.figure(figsize=(5, 4))
    plt.plot(this_X[:, 0], this_y, 'b+')

    for name, estimator in estimators:
        model = make_pipeline(PolynomialFeatures(3), estimator)
        model.fit(this_X, this_y)
        mse = mean_squared_error(model.predict(X_test), y_test)
        y_plot = model.predict(x_plot[:, np.newaxis])
        plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],
                 linewidth=lw, label='%s: error = %.3f' % (name, mse))

    legend_title = 'Error of Mean\nAbsolute Deviation\nto Non-corrupt Data'
    legend = plt.legend(loc='upper right', frameon=False, title=legend_title,
                        prop=dict(size='x-small'))
    plt.xlim(-4, 10.2)
    plt.ylim(-2, 10.2)
    plt.title(title)
plt.show()

"""
示例在不同的情况下演示了稳健拟合：
    没有测量误差，只有建模误差（用多项式拟合正弦）
    X中的测量误差
    y方向的测量误差

结论：
    1. RANSAC适用于y方向的强异常值
    2. Theilesen适用于X和y方向上的小异常值，但有一个断点，在该断点以上，它的性能比OLS差
    3. HuberRegressor的分数不能直接与TheilSen和RANSAC进行比较，因为它不完全过滤异常值，但会降低它们的影响。
    
"""

"""
算法细节：
    1. Theil-Sen回归在渐进效率和无偏估计方面足以媲美OLS。不同：Theil-Sen无参数，即没有对底层数据进行分布假设
    2. Theil-Sen基于中值的估计，更适合于离散值。
    3. 可以容忍29.3的损坏数据。

"""

