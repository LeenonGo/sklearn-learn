# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/16 16:45
# @Function:示例 用KBINS离散化器离散连续特征
# 
# 该示例比较了线性回归（线性模型）和决策树（基于树的模型）的预测结果，以及是否对实值特征进行离散化。
# 如离散化前的结果所示，线性模型构建速度快，解释相对简单，但只能建模线性关系，而决策树可以构建更复杂的数据模型。
# 使线性模型对连续数据更有效的一种方法是使用离散化（也称为装箱）。

# 在本例中，我们对特征进行离散化，并对转换后的数据进行热编码。
# 请注意，如果箱子不够宽，则过度装配的风险会明显增加，因此离散化器参数通常应在交叉验证下进行调整。

# 离散化后，线性回归和决策树做出完全相同的预测。由于每个箱子内的特征是恒定的，因此任何模型都必须为箱子内的所有点预测相同的值。
# 与离散化前的结果相比，线性模型变得更加灵活，而决策树变得更加不灵活。
# 请注意，装箱功能通常对基于树的模型没有任何好处，因为这些模型可以学习在任何地方拆分数据。
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeRegressor


# construct the dataset
rnd = np.random.RandomState(42)
X = rnd.uniform(-3, 3, size=100)
y = np.sin(X) + rnd.normal(size=len(X)) / 3
X = X.reshape(-1, 1)

# transform the dataset with KBinsDiscretizer
enc = KBinsDiscretizer(n_bins=10, encode='onehot')
X_binned = enc.fit_transform(X)

# predict with original dataset
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 4))
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
ax1.plot(line, reg.predict(line), linewidth=2, color='green', label="linear regression")
reg = DecisionTreeRegressor(min_samples_split=3, random_state=0).fit(X, y)
ax1.plot(line, reg.predict(line), linewidth=2, color='red', label="decision tree")
ax1.plot(X[:, 0], y, 'o', c='k')
ax1.legend(loc="best")
ax1.set_ylabel("Regression output")
ax1.set_xlabel("Input feature")
ax1.set_title("Result before discretization")

# predict with transformed dataset
line_binned = enc.transform(line)
reg = LinearRegression().fit(X_binned, y)
ax2.plot(line, reg.predict(line_binned), linewidth=2, color='green', linestyle='-', label='linear regression')
reg = DecisionTreeRegressor(min_samples_split=3, random_state=0).fit(X_binned, y)
ax2.plot(line, reg.predict(line_binned), linewidth=2, color='red', linestyle=':', label='decision tree')
ax2.plot(X[:, 0], y, 'o', c='k')
ax2.vlines(enc.bin_edges_[0], *plt.gca().get_ylim(), linewidth=1, alpha=.2)
ax2.legend(loc="best")
ax2.set_xlabel("Input feature")
ax2.set_title("Result after discretization")

plt.tight_layout()
plt.show()





