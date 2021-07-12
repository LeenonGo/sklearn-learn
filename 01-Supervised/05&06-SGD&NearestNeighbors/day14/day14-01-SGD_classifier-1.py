# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/12 10:37
# @Function:
# 随机梯度下降的分类

from sklearn.linear_model import SGDClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X, y)
# 预测
print(clf.predict([[2., 2.]]))

# 模型参数
print(clf.coef_)
# 截距（偏差）
print(clf.intercept_)
# 获得到此超平面的符号距离
print(clf.decision_function([[2., 2.]]))


print("-----------------损失函数----------------------")
# 损失函数：hinge、modified_huber、log
# hinge、modified_huber是懒惰的，只有一个例子违反了边界约束，才更新模型的参数.这使得训练非常有效率
# 使用 loss="log" 或者 loss="modified_huber" 来启用 predict_proba 方法
clf = SGDClassifier(loss="log").fit(X, y)
print(clf.predict_proba([[1., 1.]]))


print("-----------------惩罚项----------------------")
# 通过 penalty 参数来设定惩罚项L1、L2(默认)、elasticnet


# SGDClassifier 通过利用 “one versus all” （OVA）方法来组合多个二分类器，从而实现多分类。
# 示例：day14-01-SGD_classifier-1-eg2.py

