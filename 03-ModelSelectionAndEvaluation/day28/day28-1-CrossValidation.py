# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/10 16:25
# @Function: 3.1. 交叉验证：评估估算器的表现  https://www.scikitlearn.com.cn/0.21.3/30/
#

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets
from sklearn import svm, metrics

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print(clf.score(X_test, y_test))

# 当评价估计器的不同设置（”hyperparameters(超参数)”）时(如C参数)
# 最基本的方法被称之为，k-折交叉验证


# 计算交叉验证的指标
#  k-折交叉验证将训练集划分为 k 个较小的集合（其他方法会在下面描述，主要原则基本相同）。 每一个 k 折都会遵循下面的过程：
#
# 将 k-1 份训练集子集作为 training data （训练集）训练模型，
# 将剩余的 1 份训练集子集用于模型验证（也就是把它当做一个测试集来计算模型的性能指标，例如准确率）。
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_macro')

# cross_validate 函数与 cross_val_score 在下面的两个方面有些不同 -
#   它允许指定多个指标进行评估.
#   除了测试得分之外，它还会返回一个包含训练得分，拟合次数， score-times （得分次数）的一个字典。


# --------------------------3.1.2. 交叉验证迭代器-----------------------------

# 1. K 折
# KFold 将所有的样例划分为 k 个组，都具有相同的大小.称为折叠
# 预测函数学习时使用 k - 1 个折叠中的数据，最后一个剩下的折叠会用于测试。
import numpy as np
from sklearn.model_selection import KFold
X = ["a", "b", "c", "d"]
kf = KFold(n_splits=2)
for train, test in kf.split(X):
    print("%s  %s" % (train, test))

# RepeatedKFold 重复 K-Fold n 次。当需要运行时可以使用它 KFold n 次，在每次重复中产生不同的分割。
# RepeatedStratifiedKFold 在每个重复中以不同的随机化重复 n 次分层的 K-Fold 。
# LeaveOneOut (或 LOO) 是一个简单的交叉验证。每个学习集都是通过除了一个样本以外的所有样本创建的，测试集是被留下的样本。
# 留 P 交叉验证 (LPO) 与 LeaveOneOut 非常相似，因为它通过从整个集合中删除 p 个样本来创建所有可能的 训练/测试集。
# 随机排列交叉验证 Shuffle & Split 生成一个用户给定数量的独立的训练/测试数据划分。样例首先被打散然后划分为一对训练测试集合。


# ***************  3.1.2.2. 基于类标签、具有分层的交叉验证迭代器  *********************
# 一些分类问题在目标类别的分布上可能表现出很大的不平衡性：例如，可能会出现比正样本多数倍的负样本。
# 在这种情况下，建议采用如 StratifiedKFold 和 StratifiedShuffleSplit 中实现的分层抽样方法，
# 确保相对的类别频率在每个训练和验证 折叠 中大致保留。
# 1. 分层 k 折
#       StratifiedKFold是k-fold的变种，会返回stratified（分层）的折叠：每个小集合中，各个类别的样例比例大致和完整数据集中相同。
# 2. 分层随机 Split
#     ShuffleSplit 的一个变种，会返回直接的划分，比如： 创建一个划分，但是划分中每个类的比例和完整数据集中的相同。
# 3. 用于分组数据的交叉验证迭代器
#     如果潜在的生成过程产生依赖样本的groups ，那么i.i.d.假设将会被打破。
#          组 k-fold  留一组交叉验证  留 P 组交叉验证  Group Shuffle Split
