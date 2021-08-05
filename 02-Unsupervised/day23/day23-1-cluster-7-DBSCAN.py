# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/5 9:59
# @Function: https://www.scikitlearn.com.cn/0.21.3/22/#237-dbscan
# DBSCAN 算法将簇视为被低密度区域分隔的高密度区域。
# 与K-means相反，k-means假设簇是凸的，而DBscan发现簇可以是任何形状的。
# DBSCAN 的核心概念是 core samples, 是指位于高密度区域的样本。
# 因此一个簇是一组核心样本，每个核心样本彼此靠近（通过某个距离度量测量） 和一组接近核心样本的非核心样本（但本身不是核心样本）。
#
# 算法中的两个参数, min_samples 和 eps,正式的定义了我们所说的 稠密（dense）。
# 较高的 min_samples 或者较低的 eps 都表示形成簇所需的较高密度。

# 定义：核心样本是指数据集中的一个样本的 eps 距离范围内，存在 min_samples 个其他样本，这些样本被定义为核心样本的邻居(neighbors)。
# 根据定义，任何核心样本都是簇的一部分，任何不是核心样本并且和任意一个核心样本距离都大于eps 的样本将被视为异常值。

# 在下面的示例中，颜色表示簇成员属性，大圆圈表示算法发现的核心样本。较小的圈子表示仍然是簇的一部分的非核心样本。
# 此外，异常值由下面的黑点表示。
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)

X = StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)  # 修改这里的参数对比查看结果
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))  # 同质性
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))  # 完整性
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]  # 核心样本
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]  # 非核心样本
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

# DBSCAN 算法具有确定性的，当以相同的顺序给出相同的数据时总是形成相同的簇。
# 首先，即使核心样本总是被分配给相同的簇，这些簇的标签将取决于数据中遇到这些样本的顺序。
# 第二个更重要的是，非核心样本的簇可能因数据顺序而有所不同。 当一个非核心样本距离两个核心样本的距离都小于 eps 时，就会发生这种情况。
# 通过三角不等式可知，这两个核心样本距离一定大于 eps 或者处于同一个簇中。
# 非核心样本将被分配到首先查找到该样本的簇，因此结果也将取决于数据的顺序。







