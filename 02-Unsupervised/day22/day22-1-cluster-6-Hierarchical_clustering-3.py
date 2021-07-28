# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/28 18:10
# @Function: 改变度量标准  https://www.scikitlearn.com.cn/0.21.3/22/#2363-varying-the-metric
#
# Single, verage and complete linkage 可以使用各种距离 (or affinities),
# 特别是欧氏距离 (l2 距离), 曼哈顿距离（l1 距离), 余弦距离(cosine distance),
# 或者任何预先计算的关联矩阵(affinity matrix).
#
# l1 距离有利于稀疏特征或者稀疏噪声: 例如很多特征都是0，
# 余弦 距离是非常有趣的，因为它对全局放缩是一样的。
#
# 选择度量标准的方针是使得不同类样本之间距离最大化，并且最小化同类样本之间的距离。


# 示例：不同度量的凝聚聚类

# 演示不同度量对层次聚类的影响。
# 该示例旨在展示不同度量选择的效果。它适用于波形，可以看作是高维向量。
# 事实上，度量之间的差异通常在高维上更为明显（特别是对于l1和l2）。

# 我们从三组波形中生成数据。
# 两个波形（波形1和波形2）彼此成比例。
# 余弦距离对数据的缩放是不变的，因此，它不能区分这两种波形。
# 因此，即使没有噪声，使用该距离的聚类也不会分离出波形1和波形2。

# 我们在这些波形中加入观测噪声。我们生成非常稀疏的噪声：只有6%的时间点包含噪声。
# 因此，该噪声的l1范数（即“城市块”距离）远小于其l2范数（“欧几里德”距离）。
# 这可以在类间距离矩阵上看到：表示类的扩展的对角线上的值对于欧几里德距离比对于城市块距离要大得多。

# 当我们对数据进行聚类时，我们发现聚类反映了距离矩阵中的内容。
# 事实上，对于欧几里德距离，由于噪声的存在，类的分离是不好的，因此聚类不能分离波形。
# 对于城市街区距离，分离良好，波形类别恢复。
# 最后，余弦距离在所有波形1和波形2上都不分离，因此聚类将它们放在同一个聚类中。


import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

np.random.seed(0)

# Generate waveform data
n_features = 2000
t = np.pi * np.linspace(0, 1, n_features)


def sqr(x):
    return np.sign(np.cos(x))


X = list()
y = list()
for i, (phi, a) in enumerate([(.5, .15), (.5, .6), (.3, .2)]):
    for _ in range(30):
        phase_noise = .01 * np.random.normal()
        amplitude_noise = .04 * np.random.normal()
        additional_noise = 1 - 2 * np.random.rand(n_features)
        # Make the noise sparse
        additional_noise[np.abs(additional_noise) < .997] = 0
        d = 12 * ((a + amplitude_noise) * (sqr(6 * (t + phi + phase_noise))) + additional_noise)
        X.append(d)
        y.append(i)

# 对于数据 X  90*2000
# 前30个样本为Waveform 1 中间30为Waveform 2，后30个为Waveform 3
X = np.array(X)
y = np.array(y)

n_clusters = 3

labels = ('Waveform 1', 'Waveform 2', 'Waveform 3')

# -------------------------- 0 -----------------------------------
# Plot the ground-truth labelling
# plt.figure()
# plt.axes([0, 0, 1, 1])
# plt.plot(X[y == 0].T, c='r', alpha=.5)  # 单独显示Waveform 1
# plt.plot(X[y == 1].T, c='g', alpha=.5)  # 单独显示Waveform 2
plt.plot(X[y == 2].T, c='b', alpha=.5)  # 单独显示Waveform 3
# -------------------------- 0 -----------------------------------


# -------------------------- 1 -----------------------------------
# for l, c, n in zip(range(n_clusters), 'rgb', labels):
#     lines = plt.plot(X[y == l].T, c=c, alpha=.5)
#     lines[0].set_label(n)
#
# plt.legend(loc='best')
# plt.axis('tight')
# plt.axis('off')
# plt.suptitle("Ground truth", size=20)
# -------------------------- 1 -----------------------------------


# -------------------------- 2 -----------------------------------
# # Plot the distances
# for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):
#     avg_dist = np.zeros((n_clusters, n_clusters))
#     plt.figure(figsize=(5, 4.5))
#     for i in range(n_clusters):
#         for j in range(n_clusters):
#             avg_dist[i, j] = pairwise_distances(X[y == i], X[y == j], metric=metric).mean()
#     avg_dist /= avg_dist.max()
#     for i in range(n_clusters):
#         for j in range(n_clusters):
#             plt.text(i, j, '%5.3f' % avg_dist[i, j], verticalalignment='center', horizontalalignment='center')
#
#     # plt.imshow(avg_dist, interpolation='nearest', cmap=plt.cm.gnuplot2, vmin=0)
#     plt.xticks(range(n_clusters), labels, rotation=45)
#     plt.yticks(range(n_clusters), labels)
#     plt.colorbar()
#     plt.suptitle("Interclass %s distances" % metric, size=18)
#     plt.tight_layout()

# -------------------------- 2 -----------------------------------

# # Plot clustering results
# for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):
#     model = AgglomerativeClustering(n_clusters=n_clusters, linkage="average", affinity=metric)
#     model.fit(X)
#     plt.figure()
#     plt.axes([0, 0, 1, 1])
#     for l, c in zip(np.arange(model.n_clusters), 'rgbk'):
#         plt.plot(X[model.labels_ == l].T, c=c, alpha=.5)
#     plt.axis('tight')
#     plt.axis('off')
#     plt.suptitle("AgglomerativeClustering(affinity=%s)" % metric, size=20)

plt.show()
