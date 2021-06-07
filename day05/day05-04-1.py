# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/6/7 17:13
# @Function:  当用主成分分析(PCA)来 transform（转换） 数据时，可以通过在子空间上投影来降低数据的维数。
# 
import numpy as np
from sklearn import decomposition

x1 = np.random.normal(size=100)
x2 = np.random.normal(size=100)
x3 = x1 + x2
X = np.c_[x1, x2, x3]

pca = decomposition.PCA()
pca.fit(X)
print(pca.explained_variance_)  # 方差值。方差值越大，则说明越是重要的主成分。
print(pca.explained_variance_ratio_)  # 降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分。

print("-------------------------------------------")
pca.n_components = 2  # 指定希望PCA降维后的特征维度数目
X_reduced = pca.fit_transform(X)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
# 根据输出结果可以看出 模型抛弃了第三个特征
# print(X_reduced.shape)  # (100, 2)

print("-------------------------------------------")
# 可以指定主成分至少占比
pca = decomposition.PCA(n_components=0.70)
pca.fit(X)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.n_components_)

print("-------------------------------------------")
pca = decomposition.PCA(n_components=0.99)
pca.fit(X)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.n_components_)
# 这时候需要两者一起可以满足占比

print("-------------------------------------------")
pca = decomposition.PCA(n_components="mle")
pca.fit(X)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.n_components_)
# 使用MLE算法自动选择降维维度的效果




