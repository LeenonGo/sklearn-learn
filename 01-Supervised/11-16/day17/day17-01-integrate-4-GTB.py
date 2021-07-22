# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/20 15:54
# @Function: Gradient Tree Boosting 或梯度提升回归树（GBRT)
# 对于任意的可微损失函数的提升算法的泛化。
# GBRT 是一个准确高效的现有程序， 它既能用于分类问题也可以用于回归问题。
# 梯度树提升模型被应用到各种领域，包括网页搜索排名和生态领域。
# GBRT 的优点:
#   对混合型数据的自然处理（异构特征）
#   强大的预测能力
#   在输出空间中对异常点的鲁棒性(通过具有鲁棒性的损失函数实现)
# GBRT 的缺点:
#   可扩展性差(此处的可扩展性特指在更大规模的数据集/复杂度更高的模型上使用的能力
#


# 这个例子演示了从弱预测模型集合生成预测模型的梯度提升。
# 梯度增强可以用于回归和分类问题。在这里，我们将训练一个模型来处理糖尿病回归任务。

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)  # 分割数据

params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,  # 分割内部节点所需的最小样本数。
          'learning_rate': 0.01,  # 每棵树的贡献会减少多少。
          'loss': 'ls'  # 损失函数优化。在这种情况下使用最小二乘函数
          }

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

# 将首先计算测试集偏差，然后将其与boosting迭代进行比较
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = reg.loss_(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-', label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-', label='Test Set Deviance')

plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()


# 特征
feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(diabetes.feature_names)[sorted_idx])
plt.title('Feature Importance (MDI)')

result = permutation_importance(reg, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(diabetes.feature_names)[sorted_idx])
plt.title("Permutation Importance (test set)")
fig.tight_layout()


plt.show()
