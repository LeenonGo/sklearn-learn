# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/28 16:54
# @Function: 添加连接约束 https://www.scikitlearn.com.cn/0.21.3/22/#2362
# AgglomerativeClustering 中有一个特点：
# 可以使用连接矩阵(connectivity matrix)将连接约束添加到算法中（只有相邻的聚类可以合并到一起），
# 连接矩阵为每一个样本给定了相邻的样本。
# 例如，在 swiss-roll 的例子中，连接约束禁止在不相邻的 swiss roll 上合并，从而防止形成在 roll 上重复折叠的聚类。


# 示例：层次聚类：结构化与非结构化ward

# 示例构建了一个swiss roll数据集，并在其位置上运行层次聚类。
# 在第一步中，层次聚类不受结构的连通性约束，仅基于距离，
# 而在第二步中，聚类仅限于k近邻图：这是一种结构优先的层次聚类。

# 在没有连通性约束的情况下，学习到的一些簇不考虑swiss roll的结构，并延伸到流形的不同褶皱。
# 相反，当连接约束相反时，簇会形成一个漂亮的swiss roll。

import time as time
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import kneighbors_graph

# #############################################################################
n_samples = 1500
noise = 0.05
X, _ = make_swiss_roll(n_samples, noise=noise)
# Make it thinner
X[:, 1] *= .5

# #############################################################################
# Compute clustering
print("Compute unstructured hierarchical clustering...")
st = time.time()
ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X)
elapsed_time = time.time() - st
label = ward.labels_
print("Elapsed time: %.2fs" % elapsed_time)
print("Number of points: %i" % label.size)

# #############################################################################
# Plot result
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ids = label == l
    ax.scatter(X[ids, 0], X[ids, 1], X[ids, 2],
               color=plt.cm.jet(float(l) / np.max(label + 1)), s=20, edgecolor='k')
plt.title('Without connectivity constraints (time %.2fs)' % elapsed_time)


# #############################################################################
# 定义数据的结构。这里有10个最近的邻居
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

# #############################################################################
# Compute clustering
print("Compute structured hierarchical clustering...")
st = time.time()
ward = AgglomerativeClustering(n_clusters=6, connectivity=connectivity, linkage='ward').fit(X)
elapsed_time = time.time() - st
label = ward.labels_
print("Elapsed time: %.2fs" % elapsed_time)
print("Number of points: %i" % label.size)

# #############################################################################
# Plot result
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
               color=plt.cm.jet(float(l) / np.max(label + 1)), s=20, edgecolor='k')
plt.title('With connectivity constraints (time %.2fs)' % elapsed_time)

plt.show()


