# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/11 21:35
# @Function: 3.5. 验证曲线: 绘制分数以评估模型 https://www.scikitlearn.com.cn/0.21.3/34/
#
# 估计值的 偏差 是不同训练集的平均误差。
# 估计值的 方差 用来表示它对训练集的变化有多敏感。
# 噪声是数据的一个属性。

# 示例：
#   这个例子演示了欠拟合和过拟合的问题，以及我们如何使用具有多项式特征的线性回归来逼近非线性函数。
#   图中显示了我们想要近似的函数，它是余弦函数的一部分。
#   此外，还显示了实函数的样本和不同模型的近似值。这些模型具有不同程度的多项式特征。
#   我们可以看到，线性函数（1次多项式）不足以拟合训练样本。这被称为欠拟合。
#   4次多项式几乎完美地逼近真函数。但是，对于更高的阶数，模型将过度拟合训练数据，即，它学习训练数据的噪声。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def true_fun(X):
    return np.cos(1.5 * np.pi * X)


np.random.seed(0)

n_samples = 30
degrees = [1, 4, 15]

X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10)

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(degrees[i], -scores.mean(), scores.std()))
plt.show()
