# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/6/7 12:51
# @Function: grid_to_graph方法深入学习

# grid_to_graph方法至少接收两个参数n_x, n_y。
# 向下一层调用_to_graph方法
import numpy as np
from scipy import sparse

"""
_to_graph方法首先使用
edges = _make_edges_3d(n_x, n_y, n_z)
返回三维图像的边列表
先使用_make_edges_3d的源码查看这个方法的目的
"""

"""
该文件是对方法的简化说明
具体可参照个人博客：https://blog.csdn.net/weixin_35737303/article/details/117659448
"""

n_x = 3
n_y = 2
vertices = np.arange(n_x * n_y).reshape((n_x, n_y))
edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
edges = np.hstack((edges_right, edges_down))
# print(edges)
# [[0 2 4 0 1 2 3]
#  [1 3 5 2 3 4 5]]

n_voxels = n_x * n_y  # 6
weights = np.ones(edges.shape[1])  # [1. 1. 1. 1. 1. 1. 1.]
diag = np.ones(n_voxels)  # [1. 1. 1. 1. 1. 1.]

diag_idx = np.arange(n_voxels)  # [0 1 2 3 4 5]
i_idx = np.hstack((edges[0], edges[1]))  # [0 2 4 0 1 2 3 1 3 5 2 3 4 5]
j_idx = np.hstack((edges[1], edges[0]))  # [1 3 5 2 3 4 5 0 2 4 0 1 2 3]
n1 = np.hstack((weights, weights, diag))  # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
o1 = np.hstack((i_idx, diag_idx))  # [0 2 4 0 1 2 3 1 3 5 2 3 4 5 0 1 2 3 4 5]
o2 = np.hstack((j_idx, diag_idx))  # [1 3 5 2 3 4 5 0 2 4 0 1 2 3 0 1 2 3 4 5]
# 加入diag_idx表示填充对角线
n2 = (o1, o2)
c1 = (n1, n2)
c2 = (n_voxels, n_voxels)
graph = sparse.coo_matrix(c1, c2)

print(graph.toarray())
# 对应为1的地方就是有边的地方
# print(graph)
#

