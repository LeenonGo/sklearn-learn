# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/6 18:11
# @Function: 截断奇异值分解和隐语义分析  https://www.scikitlearn.com.cn/0.21.3/24/#252
# 

# TruncatedSVD 实现了一个奇异值分解（SVD）的变体，它只计算 k 个最大的奇异值，其中 k 是用户指定的参数。

# 当截断的 SVD被应用于 term-document矩阵时，这种转换被称为 latent semantic analysis (LSA,隐语义分析)，
# 因为它将这样的矩阵转换为低维度的"语义"空间。特别地是 LSA 能够抵抗同义词和多义词的影响
# 这导致 term-document 矩阵过度稀疏，并且在诸如余弦相似性的度量下表现出差的相似性。

# TruncatedSVD 非常类似于 PCA, 但不同之处在于它应用于样本矩阵 X 而不是它们的协方差矩阵。
# 当从特征值中减去 X 的每列（每个特征per-feature）的均值时，在得到的矩阵上应用 truncated SVD 相当于 PCA 。
# 实际上，这意味着 TruncatedSVD 转换器接受 scipy.sparse 矩阵，而不需要对它们进行密集（densify），
# 因为即使对于中型大小文档的集合, 密集化 （densifying）也可能填满内存。

