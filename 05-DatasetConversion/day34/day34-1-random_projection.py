# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/17 12:22
# @Function: 5.6. 随机投影
# random_projection 模块实现了一个简单且高效率的计算方式来减少数据维度，通过牺牲一定的精度（作为附加变量）来加速处理时间及更小的模型尺寸。
# 这个模型实现了两类无结构化的随机矩阵: Gaussian random matrix 和 sparse random matrix.
# 

# 随机投影矩阵的维度和分布是受控制的，所以可以保存任意两个数据集的距离。因此随机投影适用于基于距离的方法。

