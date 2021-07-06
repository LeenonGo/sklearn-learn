# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/6/7 11:00
# @Function:
# 
# 分层聚类算法: 谨慎使用
"""
### 分层聚类算法: 谨慎使用

分层聚类算法是一种旨在构建聚类层次结构的分析方法，一般来说，实现该算法的大多数方法有以下两种：

* **Agglomerative（聚合）** - 自底向上的方法:
    初始阶段，每一个样本将自己作为单独的一个簇，聚类的簇以最小化距离的标准进行迭代聚合。
    感兴趣的簇只有少量的样本时，该方法是很合适的。如果需要聚类的簇数量很大，该方法比K_means算法的计算效率也更高。
* **Divisive（分裂）** - 自顶向下的方法:
    初始阶段，所有的样本是一个簇，当一个簇下移时，它被迭代的进行分裂。
    当估计聚类簇数量较大的数据时，该算法不仅效率低(由于样本始于一个簇，需要被递归的进行分裂)，
    而且从统计学的角度来讲也是不合适的。

#### 连接约束聚类
对于逐次聚合聚类，通过连接图可以指定哪些样本可以被聚合在一个簇。
在 scikit 中，图由邻接矩阵来表示，通常该矩阵是一个稀疏矩阵。
这种表示方法是非常有用的，例如在聚类图像时检索连接区域(有时也被称为连接要素):


"""
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt

import skimage
from skimage.data import coins
from skimage.transform import rescale

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

# these were introduced in skimage-0.14

rescale_params = {}

# #############################################################################
# Generate data
orig_coins = coins()

# Resize it to 20% of the original size to speed up the processing
# Applying a Gaussian filter for smoothing prior to down-scaling
# reduces aliasing artifacts.
smoothened_coins = gaussian_filter(orig_coins, sigma=2)


