# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/1 16:56
# @Function:
# 
import numpy as np

y = np.array([0, 1, 1, 2, 2])
X = np.array([0, 0, 0, 0, 0])
print(y != 2)  # 1,1,1,0,0
X = X[y != 2]
y = y[y != 2]
print(X)
print(y)
