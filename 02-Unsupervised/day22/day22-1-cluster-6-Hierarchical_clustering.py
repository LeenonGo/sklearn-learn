# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/28 16:16
# @Function: 层次聚类 https://www.scikitlearn.com.cn/0.21.3/22/#236
#
# 层次聚类(Hierarchical clustering)代表着一类的聚类算法，这种类别的算法通过不断的合并或者分割内置聚类来构建最终聚类。
#  聚类的层次可以被表示成树。树根是拥有所有样本的唯一聚类，叶子是仅有一个样本的聚类。

# AgglomerativeClustering 使用自下而上的方法进行层次聚类:开始是每一个对象是一个聚类， 并且聚类别相继合并在一起。
# 连接标准(linkage criteria ) 决定用于合并策略的度量:
#   1. Ward 最小化所有聚类内的平方差总和。这是一种方差最小化(variance-minimizing )的优化方向，
#       这是与k-means 的目标函数相似的优化方法，但是用 凝聚分层（agglomerative hierarchical）的方法处理。
#   2. Maximum 或 complete linkage 最小化成对聚类间最远样本距离。
#   3. Average linkage 最小化成对聚类间平均样本距离值。
#   4. Single linkage 最小化成对聚类间最近样本距离值。

# FeatureAgglomeration 使用 凝聚聚类(agglomerative clustering) 将看上去相似的 特征组合在一起，从而减少特征的数量。


