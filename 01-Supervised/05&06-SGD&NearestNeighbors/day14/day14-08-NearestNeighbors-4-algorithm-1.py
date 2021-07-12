# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/12 15:05
# @Function:
# 
"""
1. 暴力计算
    对数据集中所有成对点之间距离进行暴力计算
    对于 D 维度中的 N 个样本来说, 这个方法的复杂度是 O[D N^2]
    暴力近邻搜索通过关键字 algorithm = 'brute' 来指定

为解决暴力计算的低效问题，产生了许多树结构的解决方法。
通过有效地编码样本的 aggregate distance (聚合距离) 信息来减少所需的距离计算量
基本思想：若A点距离B点非常远，B点距离C点非常近，可知A点与C点很遥远，不需要明确计算它们的距离。
成本降低为 O[D N \log(N)] 或更低

2. K-D树   K-dimensional tree  一个二叉树结构
    关于K-D的介绍， 参见https://zhuanlan.zhihu.com/p/45346117
    当维数很大时，效率变低。即维度灾难
3. Ball树
    解决 KD 树在高维上效率低下的问题。

"""
