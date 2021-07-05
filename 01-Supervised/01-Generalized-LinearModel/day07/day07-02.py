# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/6/28 10:44
# @Function:  岭回归 Ridge： https://www.scikitlearn.com.cn/0.21.3/2/#112
# 岭回归通过对系数的大小施加惩罚来解决 普通最小二乘法 的一些问题（不满秩，共线性问题）。
# 岭系数最小化的是带罚项的残差平方和。岭回归就是在前面最小化目标函数的后面加了一个2-范数的平方。
#
#
# 绘制作为正则化函数的岭系数

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# X is the 10x10 Hilbert matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

# alpha是控制系数收缩量的复杂性参数： alpha的值越大，收缩量越大，模型对共线性的鲁棒性也更强。
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)


ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()



