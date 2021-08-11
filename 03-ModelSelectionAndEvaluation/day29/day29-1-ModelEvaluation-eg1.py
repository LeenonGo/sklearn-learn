# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/11 17:34
# @Function: 示例：用排列测试分类分数的显著性
# 本例演示了使用permutation_test_score来评估使用排列的cross-valdiated分数的显著性。

import numpy as np

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score

import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

# 再将生成一些与iris数据集中的类标签不相关的随机特征数据（即2200个特征）。
rng = np.random.RandomState(seed=0)
n_uncorrelated_features = 2200
# 使用与iris和2200特征中相同数量的样本
X_rand = rng.normal(size=(X.shape[0], n_uncorrelated_features))
# 接下来，使用原始iris数据集计算permutation_test_score，
# 该数据集强烈预测标签以及随机生成的特征和iris标签，这些特征和标签之间应该没有依赖关系。
# 我们使用SVC分类器和准确度评分在每轮评估模型。

# permutation_test_score 通过计算分类器对数据集1000种不同排列的精度生成空分布，其中特征保持不变，但标签经历不同排列。
# 这是零假设的分布，表示特征和标签之间没有依赖关系。
# 然后，将经验p值计算为获得的分数大于使用原始数据获得的分数的排列百分比。


clf = SVC(kernel='linear', random_state=7)
cv = StratifiedKFold(2, shuffle=True, random_state=0)

score_iris, perm_scores_iris, pvalue_iris = permutation_test_score(
    clf, X, y, scoring="accuracy", cv=cv, n_permutations=1000)

score_rand, perm_scores_rand, pvalue_rand = permutation_test_score(
    clf, X_rand, y, scoring="accuracy", cv=cv, n_permutations=1000)


# Original data
fig, ax = plt.subplots()

ax.hist(perm_scores_iris, bins=20, density=True)
ax.axvline(score_iris, ls='--', color='r')
score_label = f"Score on original\ndata: {score_iris:.2f}\n" f"(p-value: {pvalue_iris:.3f})"
ax.text(0.7, 260, score_label, fontsize=12)
ax.set_xlabel("Accuracy score")
_ = ax.set_ylabel("Probability")

# Random data
fig2, ax = plt.subplots()
ax.hist(perm_scores_rand, bins=20, density=True)
ax.set_xlim(0.13)
ax.axvline(score_rand, ls='--', color='r')
score_label = (f"Score on original\ndata: {score_rand:.2f}\n"
               f"(p-value: {pvalue_rand:.3f})")
ax.text(0.14, 125, score_label, fontsize=12)
ax.set_xlabel("Accuracy score")
ax.set_ylabel("Probability")
plt.show()














