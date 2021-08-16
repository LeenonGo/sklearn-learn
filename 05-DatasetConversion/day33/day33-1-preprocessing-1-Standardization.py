# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/16 16:06
# @Function: 5.3 预处理数据 https://www.scikitlearn.com.cn/0.21.3/40/
# 
# 5.3.1 标准化，也称去均值和方差按比例缩放
# 如果个别特征或多或少看起来不是很像标准正态分布(具有零均值和单位方差)，那么它们的表现力可能会较差。

# 函数 scale 为数组形状的数据集的标准化提供了一个快捷实现:
import numpy as np
from sklearn import preprocessing

X_train = np.array(
    [[1., -1., 2.],
     [2., 0., 0.],
     [0., 1., -1.]])
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)

#  StandardScaler实现了转化器的API来计算训练集上的平均值和标准偏差，以便以后能够在测试集上重新应用相同的变换。
#  因此，这个类适用于 sklearn.pipeline.Pipeline 的早期步骤:
scaler = preprocessing.StandardScaler().fit(X_train)
print(scaler)
print(scaler.mean_)
print(scaler.scale_)
print(scaler.transform(X_train))


# 缩放类对象可以在新的数据上实现和训练集相同缩放操作:
X_test = [[-1., 1., 0.]]
print(scaler.trandform(X_test))


# 1 将特征缩放至特定范围内
# 一种标准化是将特征缩放到给定的最小值和最大值之间，通常在零和一之间，或者也可以将每个特征的最大绝对值转换至单位大小。
# 可以分别使用 MinMaxScaler 和 MaxAbsScaler 实现。

# 使用这种缩放的目的包括实现特征极小方差的鲁棒性以及在稀疏矩阵中保留零元素。
# 缩放到[0, 1]的例子
X_train = np.array(
    [[1., -1., 2.],
     [2., 0., 0.],
     [0., 1., -1.]])
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
# 如果给 MinMaxScaler 提供一个明确的 feature_range=(min, max) ，完整的公式是:
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

#  MaxAbsScaler 的工作原理非常相似，但是它只通过除以每个特征的最大值将训练数据特征缩放至 [-1, 1] 范围
# 这就意味着，训练数据应该是已经零中心化或者是稀疏数据。


# 2 缩放稀疏（矩阵）数据
# 中心化稀疏(矩阵)数据会破坏数据的稀疏结构，因此很少有一个比较明智的实现方式。
# 但是缩放稀疏输入是有意义的，尤其是当几个特征在不同的量级范围时。

# MaxAbsScaler 以及 maxabs_scale 是专为缩放数据而设计的，并且是缩放数据的推荐方法。
# 但是， scale 和 StandardScaler 也能够接受 scipy.sparse 作为输入，只要参数 with_mean=False 被准确传入它的构造器。
# 否则会出现 ValueError 的错误，因为默认的中心化会破坏稀疏性，并且经常会因为分配过多的内存而使执行崩溃。
# RobustScaler 不能适应稀疏输入，但你可以在稀疏输入使用 transform 方法。


# 3 缩放有离群值的数据
# 如果数据包含许多异常值，使用均值和方差缩放可能并不是一个很好的选择。这种情况下，使用 robust_scale 以及 RobustScaler 作为替代。


# 4 核矩阵的中心化



















