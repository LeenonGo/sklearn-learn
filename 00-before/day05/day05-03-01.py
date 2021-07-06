# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/6/7 11:10
# @Function:
# 
import numpy as np
from sklearn.feature_extraction.image import grid_to_graph

# a = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# b = a.reshape((2, 4))  # 将a转换成2行4列
# print(b)
#
# c = a.reshape((-1, 1))  # 将a转换成1列，行数为-1，不管该参数
# print(c)
#
# d = a.reshape((1, -1))
# print(d)  # 将a转换成1行，列数为-1，不管该参数
#
# X = np.reshape(a, (len(a), -1))  # 将a转换成len(a)行
# print(X)


A = np.array([[[1, 2], [1, 2], [2, 3]], [[2, 4], [1, 2], [6, 3]]])  # 三维(2,3,2)
# print(A)
B = A.reshape((len(A), -1))
# print(B)
print("1----------------------")
print(A[0])
# [[1 2]
#  [1 2]
#  [2 3]]
print("2----------------------")
print(*A[0])  # 转置 [1 2] [1 2] [2 3]
print(A[0].shape)
print("3----------------------")
print(*A[0].shape)  # 3 2
print("4----------------------")
C = grid_to_graph(*A[0].shape)  # 像素连接图，如果两个元素连接则存在边 这里参数为 3  2
print(C)

