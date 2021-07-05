# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/5 13:14
# @Function: Robustness regression 稳健回归
# 适用于回归模型包含损坏数据（corrupt data）的情况，如离群点或模型中的错误。
# 相较于最小二乘，Theil-Sen 对异常值具有鲁棒性。在二维情况下，它可以容忍高达29.3%的任意损坏数据（异常值）。


# 需要注意：
#   离群值在X上还是Y上
#   离群点比例和错误的量级


# Scikit-learn提供了三种稳健回归的预测器
#   1. HuberRegressor 一般快于 RANSAC 和 Theil Sen ，除非样本数很大，即 n_samples >> n_features 。
#       这是因为 RANSAC 和 Theil Sen 都是基于数据的较小子集进行拟合。
#       但使用默认参数时， Theil Sen 和 RANSAC 可能不如 HuberRegressor 鲁棒。
#   2. RANSAC 比 Theil Sen 更快，在样本数量上的伸缩性（适应性）更好。
#   3. RANSAC 能更好地处理y方向的大值离群点（通常情况下）。
#   4. Theil Sen 能更好地处理x方向中等大小的离群点，但在高维情况下无法保证这一特点。 实在决定不了的话，请使用 RANSAC
