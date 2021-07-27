# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/27 16:11
# @Function: 示例：k-均值假设的证明
# 这个例子旨在说明k-means将产生非直观的和可能意外的簇的情况。
# 在前三个图中，输入数据不符合一些隐含的假设，即k-均值产生了不需要的聚类。
# 在最后一个图中，k-means返回直观的聚类，尽管大小不均。
#

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# ------------------Incorrect number of clusters------------------
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Incorrect Number of Blobs")
# ------------------------------------------------------------------


# ------------------Anisotropicly distributed data------------------
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

plt.subplot(222)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("Anisotropicly Distributed Blobs")
# ------------------------------------------------------------------

# ------------------Different variance ------------------
X_varied, y_varied = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

plt.subplot(223)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("Unequal Variance")
# ------------------------------------------------------------------


# ------------------Unevenly sized blobs------------------
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_filtered)

plt.subplot(224)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
plt.title("Unevenly Sized Blobs")
# ------------------------------------------------------------------


plt.show()

