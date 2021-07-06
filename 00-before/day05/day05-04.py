# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/6/7 16:53
# @Function: 分解: 将一个信号转换成多个成份并且加载 -- 主成分分析PCA
#
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# #############################################################################
# Create the data

e = np.exp(1)  # 2.718281828459045
np.random.seed(4)


def pdf(x):
    """
    gauss = norm(loc=1, scale=2)  # loc: mean 均值， scale: standard deviation 标准差
    r_1 = gauss.pdf(X)
    或
    r_2 = norm.pdf(X, loc=0, scale=2)
    计算概率密度函数
    stats.norm(scale / e).pdf(x)
    :param x:
    :return:
    """
    return 0.5 * (stats.norm(scale=0.25 / e).pdf(x) + stats.norm(scale=4 / e).pdf(x))


y = np.random.normal(scale=0.5, size=(30000))
x = np.random.normal(scale=0.5, size=(30000))
z = np.random.normal(scale=0.1, size=len(x))

density = pdf(x) * pdf(y)
pdf_z = pdf(5 * z)

density *= pdf_z

a = x + y
b = 2 * y
c = a - b + z

norm = np.sqrt(a.var() + b.var())
a /= norm
b /= norm


# #############################################################################
# Plot the figures
def plot_figs(fig_num, elev, azim):
    fig = plt.figure(fig_num, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=elev, azim=azim)

    ax.scatter(a[::10], b[::10], c[::10], c=density[::10], marker='+', alpha=.4)
    Y = np.c_[a, b, c]

    # Using SciPy's SVD, this would be:
    # _, pca_score, Vt = scipy.linalg.svd(Y, full_matrices=False)

    pca = PCA(n_components=3)
    pca.fit(Y)
    V = pca.components_.T

    x_pca_axis, y_pca_axis, z_pca_axis = 3 * V
    x_pca_plane = np.r_[x_pca_axis[:2], - x_pca_axis[1::-1]]
    y_pca_plane = np.r_[y_pca_axis[:2], - y_pca_axis[1::-1]]
    z_pca_plane = np.r_[z_pca_axis[:2], - z_pca_axis[1::-1]]
    x_pca_plane.shape = (2, 2)
    y_pca_plane.shape = (2, 2)
    z_pca_plane.shape = (2, 2)
    ax.plot_surface(x_pca_plane, y_pca_plane, z_pca_plane)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])


elev = -40
azim = -80
plot_figs(1, elev, azim)

# elev = 30
# azim = 20
# plot_figs(2, elev, azim)

plt.show()

"""
图中样本点的分布在一个方向上是非常平坦的：即三个单变量特征中的任何一个都可以有另外两个特征来表示。
主成分分析法(PCA)可以找到使得数据分布不 flat 的矢量方向(可以反映数据主要信息的特征)。
"""

