# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/12 16:51
# @Function: 降维 https://www.scikitlearn.com.cn/0.21.3/7/#1662
# NCA可用于进行监督降维。输入数据被投影到一个由最小化NCA目标的方向组成的线性子空间上。
# 可以使用参数n_components设置所需的维数。
# 示例：使用PCA  LDA 和 NCA对数据降维的成果分析
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


n_neighbors = 3
random_state = 0

X, y = datasets.load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.5, stratify=y, random_state=random_state)

dim = len(X[0])
n_classes = len(np.unique(y))

pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=random_state))
lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))
nca = make_pipeline(StandardScaler(),  NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state))

# 使用最近邻分类器来评估这些方法
knn = KNeighborsClassifier(n_neighbors=n_neighbors)

dim_reduction_methods = [('PCA', pca), ('LDA', lda), ('NCA', nca)]

# plt.figure()
for i, (name, model) in enumerate(dim_reduction_methods):
    plt.figure()
    # plt.subplot(1, 3, i + 1, aspect=1)

    model.fit(X_train, y_train)

    knn.fit(model.transform(X_train), y_train)

    acc_knn = knn.score(model.transform(X_test), y_test)

    X_embedded = model.transform(X)

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap='Set1')
    plt.title("{}, KNN (k={})\nTest accuracy = {:.2f}".format(name, n_neighbors, acc_knn))
plt.show()

