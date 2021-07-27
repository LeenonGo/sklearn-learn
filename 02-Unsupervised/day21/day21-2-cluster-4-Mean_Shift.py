# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/27 17:38
# @Function: 均值漂移 Mean Shift:  https://www.scikitlearn.com.cn/0.21.3/22/#234-mean-shift
# 
# MeanShift 算法旨在于发现一个样本密度平滑的 blobs 。
# 基于质心的算法，通过更新质心的候选位置，这些侯选位置通常是所选定区域内点的均值。
# 然后，这些候选位置在后处理阶段被过滤以消除近似重复，从而形成最终质心集合。

# 该算法不是高度可扩展的，因为在执行算法期间需要执行多个最近邻搜索。 该算法保证收敛，但是当 质心的变化较小时，算法将停止迭代。
# 通过找到给定样本的最近质心来给新样本打上标签。

# 算法自动设定聚类的数目，而不是依赖参数 带宽（bandwidth）,带宽是决定搜索区域的size的参数。
# 这个参数可以手动设置，但是如果没有设置，可以使用提供的函数 estimate_bandwidth 获取 一个估算值。


# 示例：mean-shift聚类算法的一个演示

import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

labels = ms.labels_
cluster_centers = ms.cluster_centers_

n_clusters_ = len(np.unique(labels))

print("number of estimated clusters : %d" % n_clusters_)

# #############################################################################
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
