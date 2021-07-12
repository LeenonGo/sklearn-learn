# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/9 14:22
# @Function: 使用Gram 矩阵
#
import numpy as np
from sklearn import svm
X = np.array([[0, 0], [1, 1]])
y = [0, 1]
# 线性内核计算
gram = np.dot(X, X.T)
clf = svm.SVC(kernel='precomputed')
clf.fit(gram, y)
print(clf.predict(gram))

