# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/20 15:14
# @Function: 特征重要性评估：https://www.scikitlearn.com.cn/0.21.3/12/#11125
# 
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


# 生成一个只有3个信息特性的合成数据集。
# 明确地不重新洗牌数据集，以确保信息功能将与X的三个第一列相对应。
# 将数据集拆分为培训和测试子集。
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=3, n_redundant=0,
    n_repeated=0, n_classes=2, random_state=0, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 一个随机森林分类器将被用来计算特征的重要性。
feature_names = [f'feature {i}' for i in range(X.shape[1])]
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)

# 特征重要性由拟合属性 feature_importances_ 提供
# 它们被计算为每棵树中杂质积累减少的平均值和标准差。
start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: "
      f"{elapsed_time:.3f} seconds")


forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


# 置换特征重要性克服了基于杂质的特征重要性的局限性：它们不偏向于高基数特征，并且可以在遗漏的测试集上计算。
start_time = time.time()
result = permutation_importance(forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: "
      f"{elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=feature_names)
# 全排列重要性的计算成本更高。特征被洗牌n次，并重新调整模型以估计其重要性。
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()

plt.show()

