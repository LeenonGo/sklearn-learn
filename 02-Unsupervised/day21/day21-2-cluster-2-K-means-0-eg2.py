# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/27 16:20
# @Function: 示例： 手写数字数据集的K-均值聚类演示
# 在这个例子中，我们比较了K-means在运行时和结果质量方面的各种初始化策略。
# 在已知基本事实的情况下，我们还应用不同的聚类质量度量来判断聚类标签与基本事实的拟合优度。

import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits


data, labels = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = data.shape, np.unique(labels).size

print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")


# ----------------------- 定义评估标准 ----------------------------
# 我们将首先评估基准。在这个基准测试期间，我们打算比较KMeans的不同初始化方法。我们的基准将：
# 1. 创建一个管道，使用StandardScaler缩放数据；
# 2. 对管道配件进行培训和计时；
# 3. 度量通过不同度量获得的聚类的性能。
def bench_k_means(kmeans, name, data, labels):
    """
    Benchmark to evaluate the KMeans initialization methods.
    评估KMeans初始化方法的基准。

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # 定义只需要真实标签和估计器标签的度量
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    # 轮廓分数需要完整的数据集
    results += [metrics.silhouette_score(data, estimator[-1].labels_, metric="euclidean", sample_size=300, )]

    # Show the results
    formatter_result = "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    print(formatter_result.format(*results))


# ----------------------- 运行基准-----------------------
# 比较三种方法
#   使用kmeans++的初始化。这种方法是随机的，我们将运行初始化4次；
#   随机初始化。这种方法也是随机的，我们将运行初始化4次；
#   基于PCA投影的初始化。实际上，我们将使用PCA的组件来初始化KMeans。这种方法是确定性的，一次初始化就足够了。
print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\t\tAMI\t\tsilhouette')

kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="k-means++", data=data, labels=labels)
kmeans = KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="random", data=data, labels=labels)

pca = PCA(n_components=n_digits).fit(data)
kmeans = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
bench_k_means(kmeans=kmeans, name="PCA-based", data=data, labels=labels)

print(82 * '_')

# -----------------将PCA简化数据的结果可视化------------------------
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
kmeans.fit(reduced_data)

h = .02
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired, aspect="auto", origin="lower")
plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3, color="w", zorder=10)  # 将质心绘制为白色X
plt.title("K-means clustering on the digits dataset (PCA-reduced data)\n Centroids are marked with white cross")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


