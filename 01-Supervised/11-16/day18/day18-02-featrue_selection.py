# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/22 9:37
# @Function: 2. 单变量特征选择########################


# 通过基于单变量的统计测试来选择最好的特征。
# SelectKBest 移除那些除了评分最高的 K 个特征之外的所有特征
# SelectPercentile 移除除了用户指定的最高得分百分比之外的所有特征
# 对每个特征应用常见的单变量统计测试:
#   假阳性率（false positive rate） SelectFpr,
#   伪发现率（false discovery rate） SelectFdr ,
#   或者族系误差（family wise error） SelectFwe 。
# GenericUnivariateSelect 允许使用可配置方法来进行单变量特征选择。它允许超参数搜索评估器来选择最好的单变量特征。
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target
print(X.shape)  # (150, 4)
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print(X_new.shape)  # (150, 2)
# print(X_new)  # 返回单变量的得分和 p 值
# 对于回归: f_regression , mutual_info_regression
# 对于分类: chi2(卡方) , f_classif , mutual_info_classif
