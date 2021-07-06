# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/6 17:28
# @Function: 线性判别分析  https://www.scikitlearn.com.cn/0.21.3/3/#121
# 使用线性判别分析来降维
# 通过把输入的数据投影到由最大化类之间分离的方向所组成的线性子空间，可以执行有监督降维

#
# 示例：使用鸢尾花数据集对比 LDA 和 PCA 之间的降维差异
#
# PCA 识别数据中差异最大的属性组合。
# LDA 识别导致类之间差异最大的属性。
#
# 与PCA相比，LDA是一种使用已知类标签的有监督方法。

"""
特别地，与PCA相比，LDA是一种使用已知类标签的有监督方法。
"""

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# 各成分方差百分比
print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()
