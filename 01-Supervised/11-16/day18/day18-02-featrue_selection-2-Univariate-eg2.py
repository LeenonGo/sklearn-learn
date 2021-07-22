# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/22 9:56
# @Function:
# 这个例子说明了单变量F检验统计量和交互信息之间的差异。
# 我们考虑了3个特征x_1, x_2, x_3 在[0, 1 ]上均匀分布，目标取决于如下：
# y = x_1 + sin(6 * pi * x_2) + 0.1 * N(0, 1), 也就是说第三个特征是完全无关的。
# 下面的代码绘制了y与单个x_i的依赖关系，以及单变量F检验统计和互信息的标准化值。
# 由于F-test只捕获线性相关性，因此它将x_1评为最具鉴别能力的特征。x_1的f_test分数最高
# 交互信息（mutual_info_regression）可以捕获变量之间的任何依赖关系，并且它将x_2评为最具鉴别能力的特征
# 两种方法都正确地将x_3标记为不相关。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, mutual_info_regression

np.random.seed(0)
X = np.random.rand(1000, 3)
y = X[:, 0] + np.sin(6 * np.pi * X[:, 1]) + 0.1 * np.random.randn(1000)

f_test, _ = f_regression(X, y)
f_test /= np.max(f_test)

mi = mutual_info_regression(X, y)
mi /= np.max(mi)

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.scatter(X[:, i], y, edgecolor='black', s=20)
    plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
    if i == 0:
        plt.ylabel("$y$", fontsize=14)
    plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]),
              fontsize=16)
plt.show()

