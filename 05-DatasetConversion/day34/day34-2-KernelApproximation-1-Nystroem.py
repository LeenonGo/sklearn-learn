# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/17 12:44
# @Function: 5.7.1. 内核近似的 Nystroem 方法
# Nystroem 中实现了 Nystroem 方法用于低等级的近似核。它是通过采样 kernel 已经评估好的数据。
# 默认情况下， Nystroem 使用 rbf kernel，但它可以使用任何内核函数和预计算内核矩阵.
# 使用的样本数量 - 计算的特征维数 - 由参数 n_components 给出.
# 

