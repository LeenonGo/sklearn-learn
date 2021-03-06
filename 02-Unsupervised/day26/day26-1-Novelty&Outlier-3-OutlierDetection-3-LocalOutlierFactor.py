# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/8 19:53
# @Function: Local Outlier Factor（LOF 局部离群因子）https://www.scikitlearn.com.cn/0.21.3/26/#2733-local-outlier-factor
# 针对轻度高维数据集
# LOF算法计算出反映观测异常程度的得分（称为局部离群因子）。 它测量给定数据点相对于其邻近点的局部密度偏差。
# 算法思想是检测出具有比其邻近点明显更低密度的样本。
# 实际上，局部密度从 k 个最近邻得到。 观测数据的 LOF 得分等于其 k 个最近邻的平均局部密度与其本身密度的比值：
#   正常情况预期与其近邻有着类似的局部密度，而异常数据则预计比近邻的局部密度要小得多。

# 考虑的k个近邻数（别名参数 n_neighbors ）通常选择
#   1) 大于一个聚类簇必须包含对象的最小数量，以便其它对象可以成为该聚类簇的局部离散点，并且
#   2) 小于可能成为聚类簇对象的最大数量, 减少这K个近邻成为离群点的可能性。

# LOF 算法的优点是考虑到数据集的局部和全局属性：即使在具有不同潜在密度的离群点数据集中，它也能够表现得很好。
# 问题不在于样本是如何被分离的，而是样本与周围近邻的分离程度有多大。






