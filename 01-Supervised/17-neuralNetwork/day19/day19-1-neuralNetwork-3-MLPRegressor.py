# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/23 10:17
# @Function: 回归 https://www.scikitlearn.com.cn/0.21.3/18/#1173
#
"""
MLPRegressor 类多层感知器（MLP）的实现，在使用反向传播进行训练时的输出层没有使用激活函数，
也可以看作是使用恒等函数（identity function）作为激活函数。
因此，它使用平方误差作为损失函数，输出是一组连续值。

MLPRegressor 还支持多输出回归，其中一个样本可以有多个目标值。

"""



