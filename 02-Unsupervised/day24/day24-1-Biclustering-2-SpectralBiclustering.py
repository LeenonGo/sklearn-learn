# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/6 12:51
# @Function: SpectralBiclustering(双向谱聚类)
# 假设输入的数据矩阵具有隐藏的棋盘结构。具有这种结构的矩阵的行列可能被分区，使得在笛卡尔积中的大部分双向簇的列簇和行簇是近似恒定的。

# 示例：显示如何生成棋盘矩阵和对它进行双向聚类。
# 此示例演示如何生成棋盘数据集，并使用双向谱聚类算法对其进行双聚类。

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_checkerboard
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score


n_clusters = (4, 3)
data, rows, columns = make_checkerboard(
    shape=(300, 300), n_clusters=n_clusters, noise=10, shuffle=False, random_state=0)

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")

# shuffle clusters
rng = np.random.RandomState(0)
row_idx = rng.permutation(data.shape[0])
col_idx = rng.permutation(data.shape[1])
data = data[row_idx][:, col_idx]

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Shuffled dataset")

model = SpectralBiclustering(n_clusters=n_clusters, method='log', random_state=0)
model.fit(data)
score = consensus_score(model.biclusters_, (rows[:, row_idx], columns[:, col_idx]))

print("consensus score: {:.1f}".format(score))

fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")

plt.matshow(np.outer(np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) + 1), cmap=plt.cm.Blues)
plt.title("Checkerboard structure of rearranged data")

plt.show()



