# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/5 14:02
# @Function: Huber回归  https://www.scikitlearn.com.cn/0.21.3/2/#11154-huber
# 与 Ridge 不同，Huber对于被分为异常值的样本应用了一个线性损失。
#   如果这个样品的绝对误差小于某一阈值，样品就被分为内围值。
# 不同于 TheilSenRegressor 和 RANSACRegressor，Huber没有忽略异常值的影响，反而分配给它们较小的权重

# 示例：强异常数据集上的huberregression与 Ridge
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, Ridge

rng = np.random.RandomState(0)
X, y = make_regression(n_samples=20, n_features=1, random_state=0, noise=4.0, bias=100.0)

X_outliers = rng.normal(0, 0.5, size=(4, 1))
y_outliers = rng.normal(0, 2.0, size=4)
X_outliers[:2, :] += X.max() + X.mean() / 4.
X_outliers[2:, :] += X.min() - X.mean() / 4.
y_outliers[:2] += y.min() - y.mean() / 4.
y_outliers[2:] += y.max() + y.mean() / 4.
X = np.vstack((X, X_outliers))
y = np.concatenate((y, y_outliers))
plt.plot(X, y, 'b.')

colors = ['r-', 'b-', 'y-', 'm-']

x = np.linspace(X.min(), X.max(), 7)
epsilon_values = [1.35, 1.5, 1.75, 1.9]
for k, epsilon in enumerate(epsilon_values):
    huber = HuberRegressor(alpha=0.0, epsilon=epsilon)
    huber.fit(X, y)
    coef_ = huber.coef_ * x + huber.intercept_
    plt.plot(x, coef_, colors[k], label="huber loss, %s" % epsilon)

ridge = Ridge(alpha=0.0, random_state=0, normalize=True)
ridge.fit(X, y)
coef_ridge = ridge.coef_
coef_ = ridge.coef_ * x + ridge.intercept_
plt.plot(x, coef_, 'g-', label="ridge regression")

plt.title("Comparison of HuberRegressor vs Ridge")
plt.xlabel("X")
plt.ylabel("y")
plt.legend(loc=0)
plt.show()


"""
这个例子表明，岭回归中的预测受到数据集中存在的异常值的强烈影响。
Huber回归受异常值的影响较小，因为模型使用了这些异常值的线性损失。

随着Huber回归方程参数epsilon的增加，决策函数逼近岭函数。
"""