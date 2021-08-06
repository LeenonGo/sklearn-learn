# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/1 16:56
# @Function:
# 
import numpy as np
import matplotlib.pyplot as plt

data = np.array([[1, 2], [3, 4]])
plt.matshow(data, cmap=plt.cm.Blues)

rng = np.random.RandomState(0)
print(data.shape[0])
print(data.shape[1])
row_idx = rng.permutation(data.shape[0])
col_idx = rng.permutation(data.shape[1])
data = data[row_idx][:, col_idx]
plt.matshow(data, cmap=plt.cm.Blues)

plt.show()
