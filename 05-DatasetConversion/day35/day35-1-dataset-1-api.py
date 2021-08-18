# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/18 18:01
# @Function: 6. 数据集加载工具
#

# 6.1 通用数据集 API
#     loaders 可用来加载小的标准数据集
#     fetchers 可用来下载并加载大的真实数据集

# loaders和fetchers的所有函数都返回一个字典一样的对象，里面至少包含两项:
#   1. 数组: shape为n_samples*n_features的，对应的字典key是data(20news groups数据集除外)
#   2. numpy数组: 长度为n_samples的,包含了目标值,对应的字典key是target。

# 通过将return_X_y参数设置为True，几乎所有这些函数都可以将输出约束为只包含数据和目标的元组。数据集还包含一些对DESCR描述。

