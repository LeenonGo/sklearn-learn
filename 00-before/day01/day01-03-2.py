import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

X_train = np.c_[.5, 1].T
y_train = [.5, 1]
X_test = np.c_[0, 2].T
"""
np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
"""
np.random.seed(0)

# classifiers = dict(ols=linear_model.LinearRegression())
# classifiers = dict(ridge=linear_model.Ridge(alpha=.1))
classifiers = dict(ols=linear_model.LinearRegression(), ridge=linear_model.Ridge(alpha=.1))

for name, clf in classifiers.items():
    fig, ax = plt.subplots(figsize=(4, 3))

    for _ in range(6):
        rn = np.random.normal(size=(2, 1))  # 生成正态分布随机数
        this_X = .1 * rn + X_train  # 每次产生的随机数处理训练集
        clf.fit(this_X, y_train)  # 随机数和结果训练集拟合

        ax.plot(X_test, clf.predict(X_test), color='gray')  # 画单条线  测试集和结果测试集的预测值
        ax.scatter(this_X, y_train, s=3, c='gray', marker='o', zorder=10)  # 画散点图

    clf.fit(X_train, y_train)   # 拟合训练集
    X_test_pre = clf.predict(X_test)
    ax.plot(X_test, X_test_pre, linewidth=2, color='blue')
    ax.scatter(X_train, y_train, s=30, c='red', marker='+', zorder=10)

    ax.set_title(name)
    ax.set_xlim(0, 2)
    ax.set_ylim((0, 1.6))
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    #
    fig.tight_layout()  # 自动调整

plt.show()
