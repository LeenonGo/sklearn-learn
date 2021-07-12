# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/12 13:55
# @Function: https://www.scikitlearn.com.cn/0.21.3/7/
# 找到最近邻
from sklearn.neighbors import NearestNeighbors, KDTree
import numpy as np
X = np.array(
    [
        [-1, -1],
        [-2, -1],
        [-3, -2],
        [1, 1],
        [2, 1],
        [3, 2]
    ]
)
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
print(indices)  # 找出最近邻 样本点中距离最近的点，每个点的最近邻点是其自身，距离为0
print(distances)
print(nbrs.kneighbors_graph(X).toarray())  # 相连点之间的连接情况
print("---------------------------------------------")
kdt = KDTree(X, leaf_size=30, metric='euclidean')
print(kdt.query(X, k=2, return_distance=False))

