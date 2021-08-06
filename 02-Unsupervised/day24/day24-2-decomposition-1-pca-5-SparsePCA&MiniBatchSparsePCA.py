# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/6 18:05
# @Function:  稀疏主成分分析 (SparsePCA 和 MiniBatchSparsePCA)
# https://www.scikitlearn.com.cn/0.21.3/24/#2515-sparsepca-minibatchsparsepca

# SparsePCA 是 PCA 的一个变体，目的是提取能最大程度得重建数据的稀疏分量集合。
# 小批量稀疏 PCA ( MiniBatchSparsePCA ) 是一个 SparsePCA 的变体，它速度更快但准确度有所降低。对于给定的迭代次数，通过迭代该组特征的小块来达到速度的增加。

# (PCA) 的缺点在于，通过该方法提取的成分具有独占的密度表达式，即当表示为原始变量的线性组合时，它们具有非零系数，使之难以解释。

# 在许多情况下，真正的基础分量可以被更自然地想象为稀疏向量
# 稀疏的主成分产生更节约、可解释的表达式，明确强调了样本之间的差异性来自哪些原始特征。

