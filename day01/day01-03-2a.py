"""简单学习岭回归"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
X_train = 1 / (np.arange(1, 11, 1) + np.arange(0, 10, 1).reshape(-1, 1))  # -1表示通配 这里表示行数不管，只要一列数据
Y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])

# 创建一个alpha集合，用以验证不同alpha值对预测系数结果的影响
alphas = np.logspace(-10, -2, 10)  # 等比数列  base为底数 默认为10  这里为10的-10次方到10的-2次方的等比数列
ridge = Ridge()
# 使用不同的alpha进行数据训练，保存所有训练结果的coef_
coefs = []
for alpha in alphas:
    ridge.set_params(alpha=alpha)
    ridge.fit(X_train, Y_train)
    coefs.append(ridge.coef_)
plt.figure(figsize=(12, 9))
axes = plt.subplot(111)
axes.set_xscale('log')

plt.plot(alphas, coefs)

axes.set_xlabel('alpha')
axes.set_ylabel('coefs')
plt.show()


# 实验证明：
# alpha 大于10的-8.5次方之后对预测系数影响逐渐减小至无影响

"""
岭回归：
解决特征数量比样本量多的问题
缩减算法,判断哪些特征重要或者不重要，有点类似于降维的效果
缩减算法可以看作是对一个模型增加偏差的同时减少方差

高纬统计学习中的一个解决方法是 收缩 回归系数到0：任何两个随机选择的观察值数据集都很可能是不相关的。

岭参数 alpha 越大，偏差越大，方差越小。
"""