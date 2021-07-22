# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/20 9:36
# @Function: bagging 元估计器
# https://www.scikitlearn.com.cn/0.21.3/12/#1111-bagging-meta-estimatorbagging
# 
# 在原始训练集的随机子集上构建一类黑盒估计器的多个实例，
# 然后把这些估计器的预测结果结合起来形成最终的预测结果。
# 该方法通过在构建模型的过程中引入随机性，来减少基估计器的方差。
"""
bagging分类：
    1、如果抽取的数据集的随机子集是样本的随机子集，叫做粘贴
    2、如果样本抽取是有放回的，称为 Bagging。
    3、如果抽取的数据集的随机子集是特征的随机子集，叫做随机子空间
    4、如果基估计器构建在对于样本和特征抽取的子集之上时，叫做随机补丁
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

"""
示例：
    在回归问题中，均方误差可以被分解为偏差、方差和噪声。
    偏差通过问题估计器的预测和最佳可能估计器的差别来度量平均值（贝叶斯模型）
    方差项衡量的是估计量的预测在不同情况下的可变性。
    噪声测量由于数据的可变性而引起的误差的不可约部分。
"""
n_repeat = 50  # 计算期望值的迭代次数
n_train = 50
n_test = 1000
noise = 0.1
np.random.seed(0)

# Change this for exploring the bias-variance decomposition of other
# estimators. This should work well for estimators with high variance (e.g.,
# decision trees or KNN), but poorly for estimators with low variance (e.g.,
# linear models).
estimators = [("Tree", DecisionTreeRegressor()),
              ("Bagging(Tree)", BaggingRegressor(DecisionTreeRegressor()))]

n_estimators = len(estimators)


def f(x):
    x = x.ravel()
    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)


def generate(n_samples, noise, n_repeat=1):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X)

    if n_repeat == 1:
        y = f(X) + np.random.normal(0.0, noise, n_samples)
    else:
        y = np.zeros((n_samples, n_repeat))

        for i in range(n_repeat):
            y[:, i] = f(X) + np.random.normal(0.0, noise, n_samples)

    X = X.reshape((n_samples, 1))

    return X, y


X_train = []
y_train = []

for i in range(n_repeat):  # 生成数据
    X, y = generate(n_samples=n_train, noise=noise)
    X_train.append(X)
    y_train.append(y)

X_test, y_test = generate(n_samples=n_test, noise=noise, n_repeat=n_repeat)

plt.figure(figsize=(10, 8))

# 循环比较
for n, (name, estimator) in enumerate(estimators):
    # Compute predictions
    y_predict = np.zeros((n_test, n_repeat))

    for i in range(n_repeat):
        estimator.fit(X_train[i], y_train[i])
        y_predict[:, i] = estimator.predict(X_test)

    # Bias^2 + Variance + Noise decomposition of the mean squared error
    # 分解
    y_error = np.zeros(n_test)

    for i in range(n_repeat):
        for j in range(n_repeat):
            y_error += (y_test[:, j] - y_predict[:, i]) ** 2

    y_error /= (n_repeat * n_repeat)

    y_noise = np.var(y_test, axis=1)
    y_bias = (f(X_test) - np.mean(y_predict, axis=1)) ** 2
    y_var = np.var(y_predict, axis=1)

    print("{0}: {1:.4f} (error) = {2:.4f} (bias^2) "
          " + {3:.4f} (var) + {4:.4f} (noise)".format(name,
                                                      np.mean(y_error),
                                                      np.mean(y_bias),
                                                      np.mean(y_var),
                                                      np.mean(y_noise)))

    # Plot figures
    plt.subplot(2, n_estimators, n + 1)
    plt.plot(X_test, f(X_test), "b", label="$f(x)$")  # 蓝线 最佳可能模型
    plt.plot(X_train[0], y_train[0], ".b", label="LS ~ $y = f(x)+noise$")  # 蓝点 测试集

    for i in range(n_repeat):
        # 红线 单个预测值
        if i == 0:
            plt.plot(X_test, y_predict[:, i], "r", label=r"$\^y(x)$")
        else:
            # 其他单个决策树的预测
            plt.plot(X_test, y_predict[:, i], "r", alpha=0.05)

    # 直观地看来，这里的方差项对应于单个估计器的预测光束宽度（浅红色）。
    # 方差越大，x的预测对训练集中的微小变化越敏感。

    # 偏差项对应于估计量的平均预测 青色
    plt.plot(X_test, np.mean(y_predict, axis=1), "c", label=r"$\mathbb{E}_{LS} \^y(x)$")
    # 在右上图中，平均预测值（青色）和最佳可能模型之间的差异较大（差不多在x=2的时候）

    # 因此，我们可以观察到方差很大（红色光束相当宽）的时候偏差非常低（青色和蓝色曲线彼此接近）。
    plt.xlim([-5, 5])
    plt.title(name)

    if n == n_estimators - 1:
        plt.legend(loc=(1.1, .5))

    # 单个决策树的期望均方误差的逐点分解。
    # 它确认偏差项（蓝色）较低，而方差较大（绿色）。
    # 它还说明了误差的噪声部分，正如预期的那样，它看起来是恒定的，大约为0.01。
    plt.subplot(2, n_estimators, n_estimators + n + 1)
    plt.plot(X_test, y_error, "r", label="$error(x)$")

    plt.plot(X_test, y_bias, "b", label="$bias^2(x)$"),
    # 在右下图中，偏移曲线也略高于左下图。
    plt.plot(X_test, y_var, "g", label="$variance(x)$"),
    # 右下图预测的范围较窄，这表明方差较低。总的来说，偏差-方差分解不再相同。
    plt.plot(X_test, y_noise, "c", label="$noise(x)$")

    # 这种折衷方法更适合于bagging：
    # 对数据集引导副本上的几个决策树求平均值会略微增加偏差项，
    # 但允许更大的方差减少，从而导致较低的总体均方误差（比较下图中的红色曲线）。
    #
    # bagging集成的总误差小于单个决策树的总误差，这种差异确实主要源于方差的减小。
    plt.xlim([-5, 5])
    plt.ylim([0, 0.1])

    if n == n_estimators - 1:
        plt.legend(loc=(1.1, .5))

plt.subplots_adjust(right=.75)
plt.show()
