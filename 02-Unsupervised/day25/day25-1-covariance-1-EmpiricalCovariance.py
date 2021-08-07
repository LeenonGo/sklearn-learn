# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/7 22:07
# @Function: 2.6.1. 经验协方差  https://www.scikitlearn.com.cn/0.21.3/25/#261
# 总所周知,数据集的协方差矩阵可以被经典最大似然估计(或“经验协方差”)很好地近似
# 条件是与特征数量（描述观测值的变量）相比，观测数量足够大。

# 更准确地说，样本的最大似然估计是相应的总体协方差矩阵的无偏估计。

# 样本的经验协方差矩阵可以使用该包的函数empirical_covariance计算,
# 或者使用 EmpiricalCovariance.fit 方法将对象EmpiricalCovariance 与数据样本拟合 。

# 要注意，根据数据是否聚集，结果会有所不同，所以可能需要准确地使用参数 assume_centered
# 更准确地说，如果要使用 assume_centered=False, 测试集应该具有与训练集相同的均值向量。
# 如果不是这样，两者都应该被用户聚集， 然后再使用 assume_centered=True。

