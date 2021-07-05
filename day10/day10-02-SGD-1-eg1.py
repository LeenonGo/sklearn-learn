# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/5 12:19
# @Function: 随机梯度下降的简单例子
#

from sklearn.linear_model import SGDClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
clf.fit(X, y)
print(clf.predict([[2., 2.]]))
print(clf.coef_)  # 参数
print(clf.intercept_)  # 偏差
clf.decision_function([[2., 2.]])  # 点到超平面的距离

"""
定义损失函数
loss="hinge": 平滑,线性SVM
loss="modified_huber": 平滑铰链损失
loss="log": 逻辑回归
"""
print("---------------------------------------------------")
clf = SGDClassifier(loss="log", max_iter=5).fit(X, y)
print(clf.predict_proba([[1., 1.]]))
"""
设置惩罚
penalty="l2": L2 norm penalty on coef_. 默认.
penalty="l1": L1 norm penalty on coef_. 导致稀疏解，使大多数系数为零。
penalty="elasticnet": Convex combination of L2 and L1; (1 - l1_ratio) * L2 + l1_ratio * L1.
                      解决了L1惩罚在存在高度相关属性时的一些不足。
"""



