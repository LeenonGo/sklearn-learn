# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/27 16:03
# @Function: https://www.scikitlearn.com.cn/0.21.3/22/#232-k-means
# KMeans 算法通过把样本分离成 n 个具有相同方差的类的方式来聚集数据，最小化称为惯量或簇内平方和的标准。
# 该算法需要指定簇的数量。
# 它可以很好地扩展到大量样本(large number of samples)，并已经被广泛应用于许多不同领域的应用领域。


# 惯性被认为是测量簇内聚程度的度量(measure)。 它有各种缺点:
#   惯性假设簇是凸(convex)的和各项同性(isotropic)，这并不是总是对的。它对 细长的簇或具有不规则形状的流行反应不佳。
#   惯性不是一个归一化度量(normalized metric): 我们只知道当惯量的值较低是较好的，并且零是最优的。
#       但是在非常高维的空间中，欧氏距离往往会膨胀（这就是所谓的 “维度诅咒/维度惩罚”(curse of dimensionality)）。
#       在 k-means 聚类算法之前运行诸如 PCA 之类的降维算法可以减轻这个问题并加快计算速度。
# 在 day21-2-cluster-2-K-means-0-eg1.py 中代码说明

# 步骤
# 1. 第一步是选择初始质心，最基本的方法是从 X 数据集中选择 k 个样本。
#   初始化完成后，K-means 由接下来两个步骤之间的循环组成。
#   第一步将每个样本分配到其最近的质心。
# 2. 第二步通过取分配给每个先前质心的所有样本的平均值来创建新的质心。
# 3. 计算旧的和新的质心之间的差异，并且算法重复这些最后的两个步骤，直到该值小于阈值。换句话说，算法重复这个步骤，直到质心不再显著移动。

# K-means 相当于具有小的全对称协方差矩阵的期望最大化算法

# 给定足够的时间，K-means 将总是收敛的，但这可能是局部最小。这很大程度上取决于质心的初始化。
# 因此，通常会进行几次初始化不同质心的计算。帮助解决这个问题的一种方法是 k-means++ 初始化方案

# 该算法支持样本权重功能，该功能可以通过参数sample_weight实现。该功能允许在计算簇心和惯性值的过程中，给部分样本分配更多的权重。
# 例如，给某个样本分配一个权重值2，相当于在dataset X 中增加一个该样本的拷贝。

# 存在一个参数，以允许 K-means 并行运行，称为 n_jobs。给这个参数赋予一个正值指定使用处理器的数量（默认值: 1）。
# 值 -1 使用所有可用的处理器，-2 使用全部可用处理器减一，等等。
# 并行化(Parallelization)通常以内存的代价(cost of memory)加速计算（在这种情况下，需要存储多个质心副本，每个作业(job)使用一个副本）。


