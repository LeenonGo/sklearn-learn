# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/6 21:12
# @Function: 词典学习
# 带有预计算词典的稀疏编码 https://www.scikitlearn.com.cn/0.21.3/24/#2531
# SparseCoder 对象是一个估计器 （estimator），可以用来将信号转换成一个固定的,预计算的词典内原子（atoms）的稀疏线性组合
# 

#  词典学习的所有变体实现以下变换方法，可以通过 transform_method 初始化参数进行控制:
#
# Orthogonal matching pursuit(追求正交匹配)  最精确、无偏的重建
# Least-angle regression (最小角回归)
# Lasso computed by least-angle regression(最小角度回归的Lasso 计算)
# Lasso using coordinate descent (使用坐标下降的Lasso)
# Thresholding(阈值)  速度非常快，但是不能产生精确的重建


# 示例：带有预计算词典的稀疏编码
# 将信号转换为Ricker小波的稀疏组合。本例使用SparseCoder估计器直观地比较了不同的稀疏编码方法。
# Ricker（也称为墨西哥帽或高斯函数的二阶导数）不是一个特别好的内核来表示像这样的分段常量信号。
# 因此，可以看出添加不同宽度的原子有多重要，因此它激发了学习字典以最适合您的信号类型的动机。

# 右边的字典越丰富，大小就越小，为了保持相同的数量级，将执行更重的子采样。

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import SparseCoder
from sklearn.utils.fixes import np_version, parse_version


def ricker_function(resolution, center, width):
    """
    Discrete sub-sampled Ricker (Mexican hat) wavelet
    离散次采样Ricker（墨西哥帽）小波
    """
    x = np.linspace(0, resolution - 1, resolution)
    x = ((2 / (np.sqrt(3 * width) * np.pi ** .25))
         * (1 - (x - center) ** 2 / width ** 2)
         * np.exp(-(x - center) ** 2 / (2 * width ** 2)))
    return x


def ricker_matrix(width, resolution, n_components):
    """
    Dictionary of Ricker (Mexican hat) wavelets
    Ricker（墨西哥帽）小波词典
    """
    centers = np.linspace(0, resolution - 1, n_components)
    D = np.empty((n_components, resolution))
    for i, center in enumerate(centers):
        D[i] = ricker_function(resolution, center, width)
    D /= np.sqrt(np.sum(D ** 2, axis=1))[:, np.newaxis]
    return D


resolution = 1024  # 分辨率
subsampling = 3  # subsampling factor
width = 100
n_components = resolution // subsampling

# Compute a wavelet dictionary
D_fixed = ricker_matrix(width=width, resolution=resolution, n_components=n_components)
D_multi = np.r_[tuple(ricker_matrix(width=w, resolution=resolution, n_components=n_components // 5)
                for w in (10, 50, 100, 500, 1000))]

# Generate a signal
y = np.linspace(0, resolution - 1, resolution)
first_quarter = y < resolution / 4
y[first_quarter] = 3.
y[np.logical_not(first_quarter)] = -1.

# List the different sparse coding methods in the following format:
# (title, transform_algorithm, transform_alpha,
#  transform_n_nozero_coefs, color)
estimators = [('OMP', 'omp', None, 15, 'navy'),
              ('Lasso', 'lasso_lars', 2, None, 'turquoise'), ]
lw = 2
# Avoid FutureWarning about default value change when numpy >= 1.14
lstsq_rcond = None if np_version >= parse_version('1.14') else -1

plt.figure(figsize=(13, 6))
for subplot, (D, title) in enumerate(zip((D_fixed, D_multi), ('fixed width', 'multiple widths'))):
    plt.subplot(1, 2, subplot + 1)
    plt.title('Sparse coding against %s dictionary' % title)
    # 基于固定宽度字典的稀疏编码
    # 基于多宽度字典的稀疏编码
    plt.plot(y, lw=lw, linestyle='--', label='Original signal')
    # Do a wavelet approximation
    for title, algo, alpha, n_nonzero, color in estimators:
        coder = SparseCoder(dictionary=D, transform_n_nonzero_coefs=n_nonzero, transform_alpha=alpha, transform_algorithm=algo)
        x = coder.transform(y.reshape(1, -1))
        density = len(np.flatnonzero(x))  # 返回非0
        x = np.ravel(np.dot(x, D))
        squared_error = np.sum((y - x) ** 2)
        plt.plot(x, color=color, lw=lw, label='%s: %s nonzero coefs,\n%.2f error' % (title, density, squared_error))

    # Soft thresholding debiasing
    coder = SparseCoder(dictionary=D, transform_algorithm='threshold', transform_alpha=20)
    x = coder.transform(y.reshape(1, -1))
    _, idx = np.where(x != 0)
    x[0, idx], _, _, _ = np.linalg.lstsq(D[idx, :].T, y, rcond=lstsq_rcond)
    x = np.ravel(np.dot(x, D))
    squared_error = np.sum((y - x) ** 2)
    plt.plot(x, color='darkorange', lw=lw,
             label='Thresholding w/ debiasing:\n%d nonzero coefs, %.2f error' % (len(idx), squared_error))
    plt.axis('tight')
    plt.legend(shadow=False, loc='best')
plt.subplots_adjust(.04, .07, .97, .90, .09, .2)
plt.show()

