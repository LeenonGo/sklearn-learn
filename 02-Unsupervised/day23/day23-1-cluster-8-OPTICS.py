# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/5 10:45
# @Function: https://www.scikitlearn.com.cn/0.21.3/22/#238-optics
# OPTICS 可以认为是DBSCAN算法将eps要求从一个值放宽到一个值范围的推广。
# OPTICS与DBSCAN的关键区别：
#   OPTICS算法建立了一个可达性图，它为每个样本分配了一个reachability_(可达性距离)和一个簇 ordering_属性内的点(spot);
#   这两个属性是在模型拟合时分配的，用于确定簇的成员关系。
# 如果运行OPTICS时max_eps设置为默认值inf，
# 则可以使用 cluster_optics_dbscan方法对任意给定的eps值在线性时间内重复执行DBSCAN样式的簇提取。
# 将max_eps设置为一个较低的值将导致较短的运行时间，并可以视为从每个点到找到其他潜在可达点的最大邻域半径。

# 示例：查找高密度的核心样本并从中展开簇。此示例使用生成的数据，以使簇具有不同的密度。
# OPTICS首先使用 Xi集检测方法，然后在可达性上设置特定阈值，其对应于DBSCAN。
#
# OPTICS生成的可达性距离允许在单个数据集内对进行可变密度的簇提取。
# 如图所示，结合距离可达性和数据集ordering_产生一个可达性图，
# 其中点密度在Y轴上表示，并且点被排序以使得附近点相邻。在单个值处“切割”可达性图会产生类似DBSCAN的结果
# “切割”上方的所有点都被归类为噪声，每次从左到右读取时都表示新的簇。

# 与DBSCAN相比
#
# OPTICS cluster_optics_dbscan方法和DBSCAN的结果非常相似，但并不总是相同; 具体而言，是在标记离群点和噪声点方面。
# 这部分是因为由OPTICS处理的每个密集区域的第一个样本具有大的可达性值，使得接近其区域中的其他点，因此有时将被标记为噪声而不是离群点。
# 当它们被视为被标记为离群点或噪声的候选点时，这会影响相邻点的判断。

# 对于任何单个值eps，DBSCAN的运行时间往往比OPTICS短; 但是对于不同eps值的重复运行，单次运行OPTICS可能需要比DBSCAN更少的累积运行时间。


from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data

np.random.seed(0)
n_points_per_cluster = 250

C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
X = np.vstack((C1, C2, C3, C4, C5, C6))

clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)  # xi:确定构成簇边界的可达性图上的最小陡度。

# Run the fit
clust.fit(X)


# 对任意ε执行DBSCAN提取。
labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=0.5)
labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=2)

space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# Reachability plot
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)  # 奇异值
ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Reachability (epsilon distance)')
ax1.set_title('Reachability Plot')

# OPTICS
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = X[clust.labels_ == klass]
    ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
ax2.set_title('Automatic Clustering\nOPTICS')  # 自动聚类

# DBSCAN at 0.5
colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
for klass, color in zip(range(0, 6), colors):
    Xk = X[labels_050 == klass]
    ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], 'k+', alpha=0.1)
ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')

# DBSCAN at 2.
colors = ['g.', 'm.', 'y.', 'c.']
for klass, color in zip(range(0, 4), colors):
    Xk = X[labels_200 == klass]
    ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], 'k+', alpha=0.1)
ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')

plt.tight_layout()
plt.show()



