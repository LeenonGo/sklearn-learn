# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/6/28 15:37
# @Function: 实例：Lasso和Elastic Net(弹性网络)在稀疏信号上的表现
# 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

np.random.seed(42)
n_samples, n_features = 50, 100
X = np.random.randn(n_samples, n_features)  # 生成一些稀疏数据

idx = np.arange(n_features)
coef = (-1) ** idx * np.exp(-idx / 10)
coef[10:] = 0  # 稀疏系数
y = np.dot(X, coef)

y += 0.01 * np.random.normal(size=n_samples)  # 添加噪声

n_samples = X.shape[0]
X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]


alpha = 0.1
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)


enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet)
# print(enet.coef_)  # 稀疏矩阵
# print(np.where(enet.coef_)[0])  # 不为0的值的索引
# print(enet.coef_[enet.coef_ != 0])  # 不为0的值
# plt.stem  棉棒效果

# 第一步 输出弹性网络的参数（不为0）
m, s, _ = plt.stem(np.where(enet.coef_)[0], enet.coef_[enet.coef_ != 0],
                   markerfmt='x', label='Elastic net coefficients',
                   use_line_collection=True)
plt.setp([m, s], color="#2ca02c")

# 第二步 输出lasso的参数（不为0）
m, s, _ = plt.stem(np.where(lasso.coef_)[0], lasso.coef_[lasso.coef_ != 0],
                   markerfmt='x', label='Lasso coefficients',
                   use_line_collection=True)
plt.setp([m, s], color='#ff7f0e')

# 第三步 输出原参数（不为0），因为Y是由X与coef做矩阵乘法而得，所以coef是X的系数
plt.stem(np.where(coef)[0], coef[coef != 0], label='true coefficients',
         markerfmt='bx', use_line_collection=True)
#
plt.legend(loc='best')
plt.title("Lasso $R^2$: %.3f, Elastic Net $R^2$: %.3f"
          % (r2_score_lasso, r2_score_enet))
plt.show()