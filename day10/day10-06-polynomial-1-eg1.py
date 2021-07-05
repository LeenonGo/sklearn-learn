# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/5 14:24
# @Function:
# 多项式插值
"""
此示例显示可以使用线性模型进行非线性回归，使用管道添加非线性特征。
核方法扩展了这一思想，可以产生非常高（甚至无限）维的特征空间。
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def f(x):
    """ 多项式插值逼近函数"""
    return x * np.sin(x)


x_plot = np.linspace(0, 10, 100)

x = np.linspace(0, 10, 100)
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:20])
y = f(x)

X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

colors = ['teal', 'yellowgreen', 'gold']
lw = 2
plt.plot(x_plot, f(x_plot), color='cornflowerblue', linewidth=lw, label="ground truth")
plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")

for count, degree in enumerate([3, 4, 5]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw, label="degree %d" % degree)

plt.legend(loc='lower left')

plt.show()

