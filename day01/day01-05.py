"""分类"""
# 图中显示了在这个合成数据集中，logistic回归如何使用logistic曲线将值分类为0或1，即一级或二级。

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from scipy.special import expit

# General a toy dataset:s it's just a straight line with some Gaussian noise:
xmin, xmax = -5, 5
n_samples = 100
np.random.seed(0)
X = np.random.normal(size=n_samples)  # 根据正态分布随机100个值
y = (X > 0).astype(float)  # 判断X是否大于0
X[X > 0] *= 4  # 大于0的数乘以4
r = .3 * np.random.normal(size=n_samples)
X += r  # 随机值与X相加

X = X[:, np.newaxis]  # 将一维数据转换为2维，便于fit计算

# Fit the classifier
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(X, y)  #

# and plot the result
plt.figure(1, figsize=(8, 6))
plt.clf()
plt.scatter(X.ravel(), y, color='black', zorder=20)  # 处理后的X与原X是否大于0的散点图  这里记为A处
#
# print(clf.intercept_)  # [-1.6490375]
# print(clf.coef_ )  # [[6.90838744]]
X_test = np.linspace(-5, 10, 300)  # linspace均匀取值
# X_test = np.logspace(-5, -10, 300)
loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()  # 损失函数?  这一步公式还不理解
# expit（x）= 1 /（1 + exp（-x））  也叫logistic sigmoid
# 这一步感觉像是 用X的测试集乘以权重再加上参数（补偿），之后用logistic sigmoid得出某种值来统计损失值
# 问题：clf是X和y的拟合 y是关于X的函数  X也是关于X的函数（因为在X本身上做了一些变换计算）
#         而X_test是另外随机的变量，为什么能拟合得这么标准？？？
#
#
# X_test是为了和函数拟合而“挑选”的变量。并不是巧合
# X的取值 是一个服从正态分布的随机值
# A处的函数x轴是X变换后得到的，从负到正，与X本身的走向基本同步；y轴是X正负的判断，是一个离散值（0或1）。
# 可以得出在A处散点图中的分布，拟合连线就是一个‘s'形的曲线，与expit得到的曲线相同，因此挑选expit函数计算损失函数，能得到较好的拟合效果
plt.plot(X_test, loss, color='red', linewidth=3)
#
ols = linear_model.LinearRegression()
ols.fit(X, y)  # 线性回归拟合
plt.plot(X_test, ols.coef_ * X_test + ols.intercept_, linewidth=1)
plt.axhline(.5, color='.5')
"""
plt.axhline(y=0.0, c="r", ls="--", lw=2)
y：水平参考线的出发点
c：参考线的线条颜色
ls：参考线的线条风格
lw：参考线的线条宽度
"""
#
plt.ylabel('y')
plt.xlabel('X')
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-.25, 1.25)
plt.xlim(-4, 10)
plt.legend(('Logistic Regression Model', 'Linear Regression Model', "as"),
           loc="lower right", fontsize='small')  # 按顺序给图形注释
plt.tight_layout()
plt.show()


"""
对于分类问题，线性回归不是一个好的方法：线性回归会给数据很多远离决策边界的权值。
线性回归的取值范围为负无穷到正无穷，逻辑回归的取值为0,1

将这些权值放到sigmoid 函数 参与计算，拟合到sigmoid

即逻辑回归
"""


