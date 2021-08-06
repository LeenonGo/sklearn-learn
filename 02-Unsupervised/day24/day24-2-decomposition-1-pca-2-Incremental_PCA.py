# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/6 15:26
# @Function: https://www.scikitlearn.com.cn/0.21.3/24/#2512-pca-incremental-pca
# 增量PCA (Incremental PCA)
# PCA 对象非常有用, 但 针对大型数据集的应用, 仍然具有一定的限制。 最大的限制是 PCA 仅支持批处理，这意味着所有要处理的数据必须放在主内存。
#  IncrementalPCA 对象使用不同的处理形式, 即允许部分计算以小型批处理方式处理数据的方法进行, 而得到和 PCA 算法差不多的结果。
#
# 在应用SVD之前，IncrementalPCA就像PCA一样,为每个特征聚集而不是缩放输入数据。

# IPCA使用独立于输入数据样本数量的内存量为输入数据构建低秩近似值。它仍然依赖于输入数据功能，但更改批大小允许控制内存使用。


# 此示例用于可视化检查IPCA是否能够找到数据到PCA的类似投影（到符号翻转），同时一次只处理几个样本。
# 这可以被视为一个“玩具示例”，因为IPCA用于不适合主内存的大型数据集，需要增量方法。

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA

iris = load_iris()
X = iris.data
y = iris.target

n_components = 2
ipca = IncrementalPCA(n_components=n_components, batch_size=10)
X_ipca = ipca.fit_transform(X)

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

colors = ['navy', 'turquoise', 'darkorange']

for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1], color=color, lw=2, label=target_name)

    if "Incremental" in title:
        err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
        plt.title(title + " of iris dataset\nMean absolute unsigned error " "%.6f" % err)
    else:
        plt.title(title + " of iris dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-4, 4, -1.5, 1.5])

plt.show()

