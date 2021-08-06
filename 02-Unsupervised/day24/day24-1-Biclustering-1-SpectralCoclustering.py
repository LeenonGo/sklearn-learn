# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/6 12:33
# @Function: SpectralCoclustering(联合谱聚类)
# https://www.scikitlearn.com.cn/0.21.3/23/#241-spectral-co-clustering

# 该算法找到的双向簇的值比其它的行和列更高。
# 每一个行和列都只属于一个双向簇, 所以重新分配行和列，使得分区连续显示对角线上的高值
# 算法将输入的数据矩阵看做成二分图：
#   该矩阵的行和列对应于两组顶点，每个条目对应于行和列之间的边，该算法近似的进行归一化，对图进行切割，找到更重的子图。


# 示例：如何用双向簇产生一个数据矩阵并应用。
# 此示例演示如何使用联合谱聚类算法生成数据集并对其进行双聚类。
# 使用make_biclusters创建一个小值矩阵，并使用大值植入双聚类。

# 然后对行和列进行洗牌，并将其传递给联合谱聚类算法。重新排列洗牌矩阵以使双聚类连续显示算法找到双聚类的准确度。

import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import make_biclusters
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score

data, rows, columns = make_biclusters(shape=(300, 300), n_clusters=5, noise=5, shuffle=False, random_state=0)

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")

# shuffle clusters
rng = np.random.RandomState(0)
row_idx = rng.permutation(data.shape[0])
col_idx = rng.permutation(data.shape[1])
data = data[row_idx][:, col_idx]

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Shuffled dataset")

model = SpectralCoclustering(n_clusters=5, random_state=0)
model.fit(data)
score = consensus_score(model.biclusters_, (rows[:, row_idx], columns[:, col_idx]))

print("consensus score: {:.3f}".format(score))

fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")

plt.show()

