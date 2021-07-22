# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/22 10:04
# @Function: 递归式特征消除 https://www.scikitlearn.com.cn/0.21.3/14/#1133
# 给定一个外部的估计器，可以对特征赋予一定的权重
# recursive feature elimination ( RFE ) 通过考虑越来越小的特征集合来递归的选择特征。
# 首先，评估器在初始的特征集合上面训练并且每一个特征的重要程度是通过一个 coef_ 属性 或者 feature_importances_ 属性来获得。
# 然后，从当前的特征集合中移除最不重要的特征。
# 在特征集合上不断的重复递归这个步骤，直到最终达到所需要的特征数量为止。
# RFECV 在一个交叉验证的循环中执行 RFE 来找到最优的特征数量


# 一个递归特征消除的例子，显示了一个数字分类任务中像素的相关性。
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

# 创建RFE对象并排列每个像素
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)
ranking = rfe.ranking_.reshape(digits.images[0].shape)

# Plot pixel ranking
plt.matshow(ranking, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()


