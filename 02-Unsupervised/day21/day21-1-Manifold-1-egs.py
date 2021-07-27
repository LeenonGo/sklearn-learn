# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/27 9:17
# @Function: 介绍 https://www.scikitlearn.com.cn/0.21.3/21/#221
# 高维数据集通常难以可视化。虽然,可以通过绘制两维或三维的数据来显示高维数据的固有结构，
# 但与之等效的高维图不太直观。为了促进高维数据集结构的可视化，必须以某种方式降低维度。

# 示例：手写数字流形学习

# 数字数据集上各种嵌入的图示。
# 来自sklearn.emble模块的randomtreeembedding在技术上不是流形嵌入方法，因为它学习高维表示，
# 我们在其上应用了降维方法。然而，将数据集转换成类是线性可分离的表示形式通常是有用的。

# 在本例中，t-SNE将使用PCA生成的嵌入进行初始化，这不是默认设置。它保证了嵌入的全局稳定性，即嵌入不依赖于随机初始化。

# 来自 sklearn.discriminant_analysis模块的线性判别分析和来自sklearn.neighborks模块的邻域分量分析是有监督的降维方法，
# 即它们使用提供的标签，与其他方法相反。

from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, neighbors)

digits = datasets.load_digits(n_class=6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
n_neighbors = 30


# ----------------------------------------------------------------------
# 缩放并可视化嵌入向量
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]), color=plt.cm.Set1(y[i] / 10.), fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        # 打印缩略图
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:  # 不要显示太近的点
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# ----------------------------------------------------------------------
# 绘制数字集图像
# n_img_per_row = 20
# img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
# for i in range(n_img_per_row):
#     ix = 10 * i + 1
#     for j in range(n_img_per_row):
#         iy = 10 * j + 1
#         img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.title('A selection from the 64-dimensional digits dataset')

# ----------------------------------------------------------------------
# 基于随机酉矩阵的随机二维投影
# print("Computing random projection")
# rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
# X_projected = rp.fit_transform(X)
# plot_embedding(X_projected, "Random Projection of the digits")

# ----------------------------------------------------------------------
# 前2个主成分的投影
# print("Computing PCA projection")
# t0 = time()
# X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
# plot_embedding(X_pca, "Principal Components projection of the digits (time %.2fs)" % (time() - t0))
# 流形学习可以被认为是将线性框架（如 PCA ）推广到对数据中非线性结构敏感的一次尝试。
# 虽然存在监督变量，但是典型的流形学习问题是无监督的：它从数据本身学习数据的高维结构，而不使用预定的分类。

# ----------------------------------------------------------------------
# 前2个线性判别分量的投影

# print("Computing Linear Discriminant Analysis projection")
# X2 = X.copy()
# X2.flat[::X.shape[1] + 1] += 0.01  # 使X可逆
# t0 = time()
# X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y)
# plot_embedding(X_lda, "Linear Discriminant projection of the digits (time %.2fs)" % (time() - t0))


# ----------------------------------------------------------------------
# Isomap projection of the digits dataset

# print("Computing Isomap projection")
# t0 = time()
# X_iso = manifold.Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(X)
# plot_embedding(X_iso, "Isomap projection of the digits (time %.2fs)" % (time() - t0))

# 流形学习的最早方法之一是 Isomap 算法，等距映射（Isometric Mapping）的缩写。
# Isomap 可以被视为多维缩放（Multi-dimensional Scaling：MDS）或核主成分分析（Kernel PCA）的扩展。
# Isomap 寻求一个较低维度的嵌入(在此处，可以理解为高维数据到低维数据的一种映射转换，数据间的固有结构不变化 )
# 它保持了所有点之间的原有的测地距离:指在图中连接某两个顶点的最短距离

# ----------------------------------------------------------------------
# Locally linear embedding of the digits dataset

# print("Computing LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method='standard')
# t0 = time()
# X_lle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_lle, "Locally Linear Embedding of the digits (time %.2fs)" % (time() - t0))

# 局部线性嵌入（LLE）通过保留局部邻域内的距离来寻求数据的低维投影。
# 它可以被认为是一系列的局部主成分分析在全局范围内的相互比较，找到最优的局部非线性嵌入。

# ----------------------------------------------------------------------
# Modified Locally linear embedding of the digits dataset
# 数字数据集的改进局部线性嵌入

# print("Computing modified LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method='modified')
# t0 = time()
# X_mlle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_mlle, "Modified Locally Linear Embedding of the digits (time %.2fs)" % (time() - t0))

# 局部线性嵌入（LLE）的一个众所周知的问题是正则化问题。
# 解决正则化问题的一种方法是对邻域使用多个权重向量。这就是改进型局部线性嵌入（MLLE）算法的精髓。

# ----------------------------------------------------------------------
# HLLE embedding of the digits dataset  黑塞特征映射

# print("Computing Hessian LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method='hessian')
# t0 = time()
# X_hlle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_hlle, "Hessian Locally Linear Embedding of the digits (time %.2fs)" % (time() - t0))

# 解决 LLE 正则化问题的另一种方法
# 在每个用于恢复局部线性结构的邻域内，它会围绕一个基于黑塞的二次型展开。
# 虽然其它的实现表明它对数据大小进行缩放的能力较差，但是 sklearn 实现了一些算法改进，使得在输出低维度时它的损耗可与其他 LLE 变体相媲美。


# ----------------------------------------------------------------------
# LTSA embedding of the digits dataset
# 局部切空间对齐

# print("Computing LTSA embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method='ltsa')
# t0 = time()
# X_ltsa = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_ltsa, "Local Tangent Space Alignment of the digits (time %.2fs)" % (time() - t0))

# 与 LLE 算法关注于保持临点距离不同，LTSA 寻求通过切空间来描述局部几何形状，
# 并（通过）实现全局最优化来对其这些局部切空间，从而得知对应的嵌入。


# ----------------------------------------------------------------------
# MDS  embedding of the digits dataset
# 多维尺度分析

# print("Computing MDS embedding")
# clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
# t0 = time()
# X_mds = clf.fit_transform(X)
# print("Done. Stress: %f" % clf.stress_)
# plot_embedding(X_mds, "MDS embedding of the digits (time %.2fs)" % (time() - t0))

# 寻求数据的低维表示，而这些低维数据间的距离保持了它们在初始高维空间中的距离。
# （MDS）是一种用来分析在几何空间距离相似或相异数据的技术。MDS 尝试在几何空间上将相似或相异的数据进行建模。
# MDS算法有2类：度量和非度量。
# 在度量 MDS 中，输入相似度矩阵源自度量(并因此遵从三角形不等式)，输出两点之间的距离被设置为尽可能接近相似度或相异度的数据。
# 在非度量版本中，算法尝试保持距离的控制，并因此寻找在所嵌入空间中的距离和相似/相异之间的单调关系。
# 具体示例见


# ----------------------------------------------------------------------
# Random Trees embedding of the digits dataset

# print("Computing Totally Random Trees embedding")
# hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5)
# t0 = time()
# X_transformed = hasher.fit_transform(X)
# pca = decomposition.TruncatedSVD(n_components=2)
# X_reduced = pca.fit_transform(X_transformed)
# plot_embedding(X_reduced, "Random forest embedding of the digits (time %.2fs)" % (time() - t0))


# ----------------------------------------------------------------------

# Spectral embedding of the digits dataset
#  谱嵌入

# print("Computing Spectral embedding")
# embedder = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
# t0 = time()
# X_se = embedder.fit_transform(X)
# plot_embedding(X_se, "Spectral embedding of the digits (time %.2fs)" % (time() - t0))

# 谱嵌入是计算非线性嵌入的一种方法。


# ----------------------------------------------------------------------
# t-SNE embedding of the digits dataset
# t 分布随机邻域嵌入

# print("Computing t-SNE embedding")
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
# t0 = time()
# X_tsne = tsne.fit_transform(X)
# plot_embedding(X_tsne, "t-SNE embedding of the digits (time %.2fs)" % (time() - t0))

# t-SNE（ TSNE ）将数据点的相似性转换为概率。原始空间中的相似性表示为高斯联合概率，嵌入空间中的相似性表示为 “学生” 的 t 分布。
# 这允许 t-SNE 对局部结构特别敏感，并且有超过现有技术的一些其它优点:
#   在一个单一映射上按多种比例显示结构
#   显示位于多个、不同的流形或聚类中的数据
#   减轻在中心聚集的趋势


# ----------------------------------------------------------------------
# NCA projection of the digits dataset

# print("Computing NCA projection")
# nca = neighbors.NeighborhoodComponentsAnalysis(init='random', n_components=2, random_state=0)
# t0 = time()
# X_nca = nca.fit_transform(X, y)
# plot_embedding(X_nca, "NCA embedding of the digits (time %.2fs)" % (time() - t0))

plt.show()
