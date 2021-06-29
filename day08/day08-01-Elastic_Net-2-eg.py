# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/6/29 15:00
# @Function: Lasso和Elastic Net示例 使用坐标下降法实现 Lasso和 Elastic Net
#
import numpy as np
from sklearn import datasets
from sklearn.linear_model import lasso_path, enet_path
import matplotlib.pyplot as plt
from itertools import cycle


X, y = datasets.load_diabetes(return_X_y=True)
X /= X.std(axis=0)  # 标准化数据，更容易设置l1_ratio参数  X.std(axis=0)计算每一列的标准差

# 计算路径
eps = 5e-3  # 值越小，路径越长。路径的长度，alpha_min / alpha_max = 5e-3
# print("使用Lass计算正则化路径...")
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps=eps, fit_intercept=False)

# print("使用positive lasso计算正则化路径...")
alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
    X, y, eps=eps, positive=True, fit_intercept=False)  # positive=True,系数强制为正数

# print("使用弹性网络计算正则化路径...")
alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=0.8, fit_intercept=False)
# 在day08-01-Elastic_Net.py中介绍过，弹性网络一种使用 L1,L2 范数作为先验正则项训练的线性回归模型
# 在day07中的学习中我们得知，Ridge（岭回归）是一个L2范数的回归，Lasso是一个带L1范数的回归
# l1_ratio是一个0-1之间的数字，它在L1和L2中缩放。当l1_ratio为1时对应Lasso

# print("Computing regularization path using the positive elastic net...")
alphas_positive_enet, coefs_positive_enet, _ = enet_path(
    X, y, eps=eps, l1_ratio=0.8, positive=True, fit_intercept=False)

plt.figure(1)
colors = cycle(['b', 'r', 'g', 'c', 'k'])

neg_log_alphas_lasso = -np.log10(alphas_lasso)
neg_log_alphas_enet = -np.log10(alphas_enet)

for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)
plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and Elastic-Net Paths')
plt.legend((l1[-1], l2[-1]), ('Lasso', 'Elastic-Net'), loc='lower left')
plt.axis('tight')

# plt.figure(2)
# neg_log_alphas_positive_lasso = -np.log10(alphas_positive_lasso)
# for coef_l, coef_pl, c in zip(coefs_lasso, coefs_positive_lasso, colors):
#     l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
#     l2 = plt.plot(neg_log_alphas_positive_lasso, coef_pl, linestyle='--', c=c)
#
# plt.xlabel('-Log(alpha)')
# plt.ylabel('coefficients')
# plt.title('Lasso and positive Lasso')
# plt.legend((l1[-1], l2[-1]), ('Lasso', 'positive Lasso'), loc='lower left')
# plt.axis('tight')
#
#
# plt.figure(3)
# neg_log_alphas_positive_enet = -np.log10(alphas_positive_enet)
# for (coef_e, coef_pe, c) in zip(coefs_enet, coefs_positive_enet, colors):
#     l1 = plt.plot(neg_log_alphas_enet, coef_e, c=c)
#     l2 = plt.plot(neg_log_alphas_positive_enet, coef_pe, linestyle='--', c=c)
#
# plt.xlabel('-Log(alpha)')
# plt.ylabel('coefficients')
# plt.title('Elastic-Net and positive Elastic-Net')
# plt.legend((l1[-1], l2[-1]), ('Elastic-Net', 'positive Elastic-Net'),
#            loc='lower left')
# plt.axis('tight')


plt.show()

# 关于坐标下降法
# https://zhuanlan.zhihu.com/p/59734411

