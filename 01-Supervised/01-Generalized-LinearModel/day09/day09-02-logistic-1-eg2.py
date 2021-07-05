# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/1 17:03
# @Function: L1罚项-logistic回归的路径
#
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import datasets
from sklearn.svm import l1_min_c

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 2]
y = y[y != 2]

X /= X.max()  # 标准化X，加速收敛

# #############################################################################
# 将C的值稀疏一下
cs = l1_min_c(X, y, loss='log') * np.logspace(0, 7, 16)  # l1_min_c返回X的最小值，保证C不为空。

print("Computing regularization path ...")
start = time()
clf = linear_model.LogisticRegression(penalty='l1', solver='liblinear',
                                      tol=1e-6, max_iter=int(1e6),
                                      warm_start=True,
                                      intercept_scaling=10000.)
coefs_ = []
for c in cs:
    clf.set_params(C=c)
    clf.fit(X, y)
    coefs_.append(clf.coef_.ravel().copy())
print("This took %0.3fs" % (time() - start))

coefs_ = np.array(coefs_)
plt.plot(cs, coefs_, marker='o')
ymin, ymax = plt.ylim()
plt.xlabel('log(C)')
plt.ylabel('Coefficients')
plt.title('Logistic Regression Path')
plt.axis('tight')
plt.show()

"""
该示例是基于鸢尾花数据集的二分类问题，训练L1惩罚的逻辑回归
首先将C的最小值稀释成16个值
然后带入C值迭代训练，得到结果权重（16*4）
画出的图横坐标是C稀释的16个值，纵坐标是每个值对应的结果权重
在图的左侧（强正则化器），所有系数都正好为0。当正则化逐渐变松时，系数可以一个接一个地获得非零值
"""
