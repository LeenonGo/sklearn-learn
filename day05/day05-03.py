# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/6/7 11:06
# @Function: 特征聚集
# 稀疏性可以缓解特征维度带来的问题.即与特征数量相比，样本数量太少
# 另一个解决该问题的方式是合并相似的维度：feature agglomeration（特征聚集）
# 该方法可以通过对特征聚类来实现。换句话说，就是对样本数据转置后进行聚类。
"""显示了如何使用特征聚集将相似的特征合并在一起。"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, cluster
from sklearn.feature_extraction.image import grid_to_graph

digits = datasets.load_digits()
images = digits.images
rows = len(images)
X = np.reshape(images, (rows, -1))  # 把images数组变成rows行
connectivity = grid_to_graph(*images[0].shape)  # 得到8*8图像的边
# print(connectivity.toarray())  # 查看 # 64*64
agglo = cluster.FeatureAgglomeration(connectivity=connectivity, n_clusters=32)
# 使用自下而上法作层次聚类：每个观测初始自成一类，然后连续地合并类。连接准则确定合并类的测度

agglo.fit(X)  # (1797, 64)
X_reduced = agglo.transform(X)  # (1797, 32)
#
X_restored = agglo.inverse_transform(X_reduced)
images_restored = np.reshape(X_restored, images.shape)
plt.figure(1, figsize=(4, 3.5))
plt.clf()
plt.subplots_adjust(left=.01, right=.99, bottom=.01, top=.91)
for i in range(4):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images[i], cmap=plt.cm.gray, vmax=16, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    if i == 1:
        plt.title('Original data')
    plt.subplot(3, 4, 4 + i + 1)
    plt.imshow(images_restored[i], cmap=plt.cm.gray, vmax=16, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    if i == 1:
        plt.title('Agglomerated data')

plt.subplot(3, 4, 10)
# print(images[0].shape)  # 8*8
print(agglo.labels_)
plt.imshow(np.reshape(agglo.labels_, images[0].shape),
           interpolation='nearest', cmap=plt.cm.nipy_spectral)
plt.xticks(())
plt.yticks(())
plt.title('Labels')
plt.show()


"""
代码的核心是
agglo = cluster.FeatureAgglomeration(connectivity=connectivity, n_clusters=32)
agglo.fit(X) 
X_reduced = agglo.transform(X)
X_restored = agglo.inverse_transform(X_reduced)

使用模型拟合数据
然后将该数据变形和逆变形

但是不太懂标题说“显示了如何使用特征聚集将相似的特征合并在一起”是什么意思
X_reduced = agglo.transform(X) 
这一步是起到了降维的效果，应该就是对“将相似的特征合并在一起”的对应

X_restored = agglo.inverse_transform(X_reduced)
这一步是对样本进行转置

对样本转置后聚类可以用来解决与特征数量相比，样本数量太少的问题。

"""



