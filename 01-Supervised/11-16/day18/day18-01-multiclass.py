# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/22 8:55
# @Function: 多类和多标签算法
# sklearn.multiclass 模块采用了 元评估器 ，通过把多类 和 多标签 分类问题分解为 二元分类问题去解决。
# 这同样适用于多目标回归问题。


# ########################多标签分类格式################################
# MultiLabelBinarizer 转换器可以用来在标签接口和格式指示器接口之间进行转换。
from sklearn.preprocessing import MultiLabelBinarizer
y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]
print(MultiLabelBinarizer().fit_transform(y))
"""
说明：
[1, 0, 0], [0, 1, 1], [0, 0, 0]
    表示第一个样本属于第 0 个标签，第二个样本属于第一个和第二个标签，第三个样本不属于任何标签。
"""

# #####################1对多#############################
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.svm import LinearSVC
iris = datasets.load_iris()
X, y = iris.data, iris.target
r = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
print(r)

# #####################1对1#############################
r2 = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
print(r2)


# #####################误差校正输出代码#############################
clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)
r3 = clf.fit(X, y).predict(X)
print(r3)
