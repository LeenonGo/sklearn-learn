# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/8 15:50
# @Function: 2.7.3.2. Isolation Forest（隔离森林）  https://www.scikitlearn.com.cn/0.21.3/26/#2732-isolation-forest
# 
# 在高维数据集中实现离群点检测的一种有效方法是使用随机森林。
# ensemble.IsolationForest 通过随机选择一个特征,然后随机选择所选特征的最大值和最小值之间的分割值来"隔离"观测。
# 由于递归划分可以由树形结构表示，因此隔离样本所需的分割次数等同于从根节点到终止节点的路径长度。
# 随机划分能为异常观测产生明显的较短路径。因此，当随机树的森林共同地为特定样本产生较短的路径长度时，这些样本就很有可能是异常的。
# 在这样的随机树的森林中取平均的路径长度是数据正态性和我们的决策功能的量度。

# from sklearn.ensemble import IsolationForest  # 2.7.4Forest
# import numpy as np
#
# # 支持warm_start=True，这让可以添加更多的树到一个已拟合好的模型中:
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [0, 0], [-20, 50], [3, 5]])
# clf = IsolationForest(n_estimators=10, warm_start=True)
# clf.fit(X)  # fit 10 trees
# clf.set_params(n_estimators=20)  # add 10 more trees
# clf.fit(X)  # fit the added trees


# 使用IsolationForest进行异常检测的示例。
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# Generate train data
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))  #

# fit the model
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green', s=20, edgecolor='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c], ["training observations", "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()
