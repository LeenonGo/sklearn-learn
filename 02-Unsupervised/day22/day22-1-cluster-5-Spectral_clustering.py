# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/28 13:34
# @Function: 谱聚类 Spectral clustering  https://www.scikitlearn.com.cn/0.21.3/22/#235-spectral-clustering
# 
# SpectralClustering 是在样本之间进行关联矩阵的低维度嵌入，然后在低维空间中使用 KMeans 算法。
# 聚类 需要指定簇的数量。这个算法适用于簇数量少时，在簇数量多时是不建议使用。
# 对于两个簇，它解决了相似图形上的 归一化切割(normalised cuts)的凸松弛问题:
# 将图形切割成两个，使得切割的边缘的权重比每个簇内的边缘的权重小。
# 在图像处理时，图像的顶点是像素，相似图形的边缘是图像的渐变函数。


# 示例：用于图像分割的谱聚类方法
# 在本例中，生成具有连接圆的图像，并使用谱聚类来分离圆。
# 在这些设置中，谱聚类方法解决了称为“归一化图切割”的问题：
#   将图像视为一个连通体素图，谱聚类算法相当于选择定义区域的图割，同时最小化沿割的梯度比和区域体积。
# 当算法试图平衡体积（即平衡区域大小）时，如果取不同大小的圆，分割就会失败。

# 另外，由于图像的强度或梯度没有任何有用的信息，我们选择对一个只受梯度影响很小的图进行谱聚类。这接近于对图进行Voronoi划分。
# 此外，我们使用对象的遮罩来限制图形到对象的轮廓。在本例中，我们感兴趣的是将对象彼此分开，而不是从背景中分开。

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

l = 100
x, y = np.indices((l, l))

center1 = (28, 24)
center2 = (40, 50)
center3 = (67, 58)
center4 = (24, 70)

radius1, radius2, radius3, radius4 = 16, 14, 15, 14

circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2

# #############################################################################
# 4 circles
img = circle1 + circle2 + circle3 + circle4  # 合成数据
# plt.matshow(img)  # 可以看出数据的原始展示

# 我们使用一个仅限于前景的遮罩：这里我们感兴趣的问题不是将对象与背景分离，而是将它们彼此分离。
mask = img.astype(bool)
img = img.astype(float)
# plt.matshow(img)
r = np.random.randn(*img.shape)  # 遮罩  plt.matshow(r)
img += 1 + 0.2 * r
# plt.matshow(img)

# 将图像转换为边缘具有梯度值的图形。
graph = image.img_to_graph(img, mask=mask)
# 取梯度的一个递减函数：我们取它弱依赖于梯度分割接近于voronoi
graph.data = np.exp(-graph.data / graph.data.std())  # 可以将这一段注释之后对比查看结果
# 强制解算器为arpack，因为在这个例子中amg在数值上是不稳定的
labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')  # 分割圆
label_im = np.full(mask.shape, -1.)
label_im[mask] = labels
plt.matshow(label_im)

# #############################################################################
# 2 circles
img = circle1 + circle2

mask = img.astype(bool)
img = img.astype(float)

img += 1 + 0.2 * np.random.randn(*img.shape)

graph = image.img_to_graph(img, mask=mask)
graph.data = np.exp(-graph.data / graph.data.std())

labels = spectral_clustering(graph, n_clusters=2, eigen_solver='arpack')
label_im = np.full(mask.shape, -1.)
label_im[mask] = labels

# plt.matshow(img)
# plt.matshow(label_im)

plt.show()
