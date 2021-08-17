# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/17 12:45
# @Function: 5.7.2. 径向基函数内核
# RBFSampler 为径向基函数核构造一个近似映射，在应用线性算法（例如线性 SVM ）之前，可以使用此转换来明确建模内核映射:
#
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
X = [[0, 0], [1, 1], [1, 0], [0, 1]]
y = [0, 0, 1, 1]
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)
clf = SGDClassifier()
clf.fit(X_features, y)
print(clf.score(X_features, y))
