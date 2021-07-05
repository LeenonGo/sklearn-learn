# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/1 12:20
# @Function: BayesianRidge：贝叶斯岭回归：https://www.scikitlearn.com.cn/0.21.3/2/#11101
#

"""
下面的示例在合成数据上计算贝叶斯岭回归


与最小二乘相比，系数的权值向0回归，从而使其更稳定
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.linear_model import BayesianRidge, LinearRegression

# #############################################################################
np.random.seed(0)
n_samples, n_features = 100, 100
X = np.random.randn(n_samples, n_features)  # 生成高斯数据
lambda_ = 4.  # 创建精度λ为4的权重
w = np.zeros(n_features)
relevant_features = np.random.randint(0, n_features, 10)
for i in relevant_features:
    w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))  # 高斯
alpha_ = 50.
noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=n_samples)
y = np.dot(X, w) + noise  # 目标

# #############################################################################
#
clf = BayesianRidge(compute_score=True)
clf.fit(X, y)

ols = LinearRegression()
ols.fit(X, y)

# #############################################################################
# Plot true weights, estimated weights, histogram of the weights, and
# predictions with standard deviations
lw = 2
# plt.figure(figsize=(6, 5))  # 权重分布图
# plt.title("Weights of the model")
# plt.plot(clf.coef_, color='lightgreen', linewidth=lw, label="Bayesian Ridge estimate")  # 贝叶斯岭回归权重
# plt.plot(w, color='gold', linewidth=lw, label="Ground truth")  # 真实权重
# plt.plot(ols.coef_, color='navy', linestyle='--', label="OLS estimate")  # 最小二乘权重
# plt.xlabel("Features")
# plt.ylabel("Values of the weights")
# plt.legend(loc="best", prop=dict(size=12))
#
# # 贝叶斯岭回归权重直方图
# plt.figure(figsize=(6, 5))
# plt.title("Histogram of the weights")
# plt.hist(clf.coef_, bins=n_features, color='gold', log=True, edgecolor='black')
# plt.scatter(clf.coef_[relevant_features], np.full(len(relevant_features), 5.),
#             color='navy', label="Relevant features")
# plt.ylabel("Features")
# plt.xlabel("Values of the weights")
# plt.legend(loc="upper left")


# # 边际对数似然
# plt.figure(figsize=(6, 5))
# plt.title("Marginal log-likelihood")
# plt.plot(clf.scores_, color='navy', linewidth=lw)
# plt.ylabel("Score")
# plt.xlabel("Iterations")


# Plotting some predictions for polynomial regression
def f(x, noise_amount):
    y = np.sqrt(x) * np.sin(x)
    noise = np.random.normal(0, 1, len(x))
    return y + noise_amount * noise


degree = 10
X = np.linspace(0, 10, 100)
y = f(X, noise_amount=0.1)
clf_poly = BayesianRidge()
clf_poly.fit(np.vander(X, degree), y)

X_plot = np.linspace(0, 11, 25)
y_plot = f(X_plot, noise_amount=0)
y_mean, y_std = clf_poly.predict(np.vander(X_plot, degree), return_std=True)
plt.figure(figsize=(6, 5))
plt.errorbar(X_plot, y_mean, y_std, color='navy',
             label="Polynomial Bayesian Ridge Regression", linewidth=lw)
plt.plot(X_plot, y_plot, color='gold', linewidth=lw,
         label="Ground Truth")
plt.ylabel("Output y")
plt.xlabel("Feature X")
plt.legend(loc="lower left")
plt.show()

