# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/17 12:25
# @Function: 5.6.1. Johnson-Lindenstrauss 辅助定理

# 在数学中，johnson - lindenstrauss 引理是一种将高维的点从高维到低维欧几里得空间的低失真嵌入的方案。
# 引理阐释了高维空间下的一小部分的点集可以内嵌到非常低维的空间，这种方式下点之间的距离几乎全部被保留。
# 有了样本数量， sklearn.random_projection.johnson_lindenstrauss_min_dim
# 会保守估计随机子空间的最小大小来保证随机投影导致的变形在一定范围内：

from sklearn.random_projection import johnson_lindenstrauss_min_dim
print(johnson_lindenstrauss_min_dim(n_samples=1e6, eps=0.5))
print(johnson_lindenstrauss_min_dim(n_samples=1e6, eps=[0.5, 0.1, 0.01]))
print(johnson_lindenstrauss_min_dim(n_samples=[1e4, 1e5, 1e6], eps=0.1))











