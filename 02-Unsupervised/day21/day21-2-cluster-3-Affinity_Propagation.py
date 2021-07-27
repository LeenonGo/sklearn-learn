# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/27 17:20
# @Function: Affinity Propagation  https://www.scikitlearn.com.cn/0.21.3/22/#233-affinity-propagation
# AffinityPropagation
#   AP聚类是通过在样本对之间发送消息直到收敛的方式来创建聚类。
#   然后使用少量模范样本作为聚类中心来描述数据集，而这些模范样本可以被认为是最能代表数据集中其它数据的样本。
#   在 样本对 之间发送的消息表示一个样本作为另一个样本的模范样本的 适合程度，适合程度值在根据通信的反馈不断更新。
#   更新迭代直到收敛，完成聚类中心的选取，因此也给出了最终聚类。

# 可以根据提供的数据决定聚类的数目
# 因此有两个比较重要的参数
#   _preference_, 决定使用多少个模范样本
#   阻尼因子(damping factor) 减少吸引信息和归属信息以减少更新这些信息时的数据振荡。
# 主要的缺点是算法的复杂度.


# 样本之间传递的信息有两种。
#   第一种是吸引信息 (responsibility)
#       r(i, k), 样本 k 适合作为样本 i 的聚类中心(exemplar,即模范样本)的程度。
#   第二种是 归属信息(availability)
#       a(i, k), 样本 i 选择样本 k 作为聚类中心的适合程度,并且考虑其他所有样本选取 k 做为聚类中心的合适程度。


# 示例：AP聚类算法演示

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle

# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=0)

# #############################################################################
# Compute Affinity Propagation
af = AffinityPropagation(preference=-50).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

# #############################################################################
# Plot result

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

