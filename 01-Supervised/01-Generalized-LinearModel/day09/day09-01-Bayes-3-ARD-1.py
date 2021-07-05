# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/1 13:04
# @Function: ARDRegression 主动相关决策理论:https://www.scikitlearn.com.cn/0.21.3/2/#11102-ard
# ARDRegression （主动相关决策理论）和 Bayesian Ridge Regression 非常相似，但是会导致一个更加稀疏的权重w
# 提出一个不同的 w 的先验假设，弱化了高斯分布为球形的假设。采用 w 分布是与轴平行的椭圆高斯分布.
# 也称为：稀疏贝叶斯学习 或 相关向量机

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import ARDRegression, LinearRegression

# #############################################################################
# 与前一个代码使用相同的合成数据
np.random.seed(0)
n_samples, n_features = 100, 100
X = np.random.randn(n_samples, n_features)
lambda_ = 4.
w = np.zeros(n_features)
relevant_features = np.random.randint(0, n_features, 10)
for i in relevant_features:
    w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
alpha_ = 50.
noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=n_samples)
y = np.dot(X, w) + noise

# #############################################################################
clf = ARDRegression(compute_score=True)
clf.fit(X, y)

ols = LinearRegression()
ols.fit(X, y)


# #############################################################################
# plt.figure(figsize=(6, 5))
# plt.title("Weights of the model")
# plt.plot(clf.coef_, color='darkblue', linestyle='-', linewidth=2,  label="ARD estimate")
# plt.plot(ols.coef_, color='yellowgreen', linestyle=':', linewidth=2, label="OLS estimate")
# plt.plot(w, color='orange', linestyle='-', linewidth=2, label="Ground truth")
# plt.xlabel("Features")
# plt.ylabel("Values of the weights")
# plt.legend(loc=1)
# #
# plt.figure(figsize=(6, 5))
# plt.title("Histogram of the weights")
# plt.hist(clf.coef_, bins=n_features, color='navy', log=True)
# plt.scatter(clf.coef_[relevant_features], np.full(len(relevant_features), 5.),
#             color='gold', marker='o', label="Relevant features")
# plt.ylabel("Features")
# plt.xlabel("Values of the weights")
# plt.legend(loc=1)
# #
# plt.figure(figsize=(6, 5))
# plt.title("Marginal log-likelihood")
# plt.plot(clf.scores_, color='navy', linewidth=2)
# plt.ylabel("Score")
# plt.xlabel("Iterations")
#
#
# Plotting some predictions for polynomial regression
def f(x, noise_amount):
    y = np.sqrt(x) * np.sin(x)
    noise = np.random.normal(0, 1, len(x))
    return y + noise_amount * noise


degree = 10
X = np.linspace(0, 10, 100)
y = f(X, noise_amount=1)
clf_poly = ARDRegression(threshold_lambda=1e5)
clf_poly.fit(np.vander(X, degree), y)

X_plot = np.linspace(0, 11, 25)
y_plot = f(X_plot, noise_amount=0)
y_mean, y_std = clf_poly.predict(np.vander(X_plot, degree), return_std=True)
plt.figure(figsize=(6, 5))
plt.errorbar(X_plot, y_mean, y_std, color='navy',
             label="Polynomial ARD", linewidth=2)
plt.plot(X_plot, y_plot, color='gold', linewidth=2,
         label="Ground Truth")
plt.ylabel("Output y")
plt.xlabel("Feature X")
plt.legend(loc="lower left")
plt.show()
