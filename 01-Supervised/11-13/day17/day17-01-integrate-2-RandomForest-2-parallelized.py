# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/20 15:09
# @Function: 并行化： https://www.scikitlearn.com.cn/0.21.3/12/#11124
# n_jobs = k ，则计算被划分为 k 个作业，并运行在机器的 k 个核上
# 如果设置 n_jobs = -1 ，则使用机器的所有核。
# 

from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import ExtraTreesClassifier
"""
这个例子展示了在图像分类任务（faces）中使用树的森林来评估基于杂质的像素重要性。像素越热，越重要。
"""
n_jobs = 1  # 调整

data = fetch_olivetti_faces()
X, y = data.data, data.target

mask = y < 5  # Limit to 5 classes
X = X[mask]
y = y[mask]

# Build a forest and compute the pixel importances
print("Fitting ExtraTreesClassifier on faces data with %d cores..." % n_jobs)
t0 = time()
forest = ExtraTreesClassifier(n_estimators=1000,
                              max_features=128,
                              n_jobs=n_jobs,
                              random_state=0)

forest.fit(X, y)
print("done in %0.3fs" % (time() - t0))
importances = forest.feature_importances_
importances = importances.reshape(data.images[0].shape)

# Plot pixel importances
plt.matshow(importances, cmap=plt.cm.hot)
plt.title("Pixel importances with forests of trees")
plt.show()
