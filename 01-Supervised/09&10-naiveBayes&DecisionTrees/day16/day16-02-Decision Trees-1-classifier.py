# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/14 9:05
# @Function: 决策树的分类问题： https://www.scikitlearn.com.cn/0.21.3/11/#1101
# 

from sklearn import tree
from sklearn.datasets import load_iris

clf = tree.DecisionTreeClassifier()

# X = [[0, 0], [1, 1]]
# Y = [0, 1]
# clf = clf.fit(X, Y)
# print(clf.predict([[2., 2.]]))  # 预测
# print(clf.predict_proba([[2., 2.]]))  # 预测每个类的概率

# 可用于二分类和多分类
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)



