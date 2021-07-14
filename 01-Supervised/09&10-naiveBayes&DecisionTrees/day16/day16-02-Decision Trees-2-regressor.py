# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/14 9:10
# @Function: 决策树的回归问题 https://www.scikitlearn.com.cn/0.21.3/11/#1102
# 示例：
"""
决策树的一维回归。
利用决策树拟合正弦曲线，并加入噪声观测。因此，它学习逼近正弦曲线的局部线性回归。
我们可以看到，如果树的最大深度（由max_depth参数控制）设置得太高，决策树将学习训练数据的太精细细节，并从噪声中学习，即过拟合。
决策树通过if-then-else的决策规则来学习数据从而估测数一个正弦图像。决策树越深入，决策规则就越复杂并且对数据的拟合越好。
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))  # 噪声

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",  c="darkorange", label="data")  # 原始数据
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

