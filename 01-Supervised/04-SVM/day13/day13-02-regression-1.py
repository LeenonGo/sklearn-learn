# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/9 12:47
# @Function: 回归 https://www.scikitlearn.com.cn/0.21.3/5/#142
# 持向量回归生成的模型只依赖于训练集的子集, 因为构建模型的 cost function 忽略任何接近于模型预测的训练数据.
# 有三种不同的实现形式: SVR, NuSVR 和 LinearSVR
# 在只考虑线性核的情况下, LinearSVR 比 SVR 提供一个更快的实现形式,
# 然而比起 SVR 和 LinearSVR, NuSVR 实现一个稍微不同的构思

