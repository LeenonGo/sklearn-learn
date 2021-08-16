# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/16 16:44
# @Function: 离散化 (Discretization) (有些时候叫 量化(quantization) 或 装箱(binning)) 提供了将连续特征划分为离散特征值的方法。
#
# 1 K-bins 离散化
# KBinsDiscretizer 类使用k个等宽的bins把特征离散化。 day33-1-preprocessing-5-Discretization-eg1.py

# 2 特征二值化
# 特征二值化 是 将数值特征用阈值过滤得到布尔值 的过程。这对于下游的概率型模型是有用的，它们假设输入数据是多值 伯努利分布(Bernoulli distribution) 。




