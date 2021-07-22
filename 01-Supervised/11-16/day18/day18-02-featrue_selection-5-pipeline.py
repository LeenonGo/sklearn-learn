# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/22 10:42
# @Function: 特征选取作为 pipeline（管道）的一部分  https://www.scikitlearn.com.cn/0.21.3/14/#1135-pipeline
# 特征选择通常在实际的学习之前用来做预处理。在 scikit-learn 中推荐的方式是使用 :sklearn.pipeline.Pipeline:
"""
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)
利用 sklearn.svm.LinearSVC 和 sklearn.feature_selection.SelectFromModel 来评估特征的重要性并且选择出相关的特征。
然后，在转化后的输出中使用一个 sklearn.ensemble.RandomForestClassifier 分类器,比如只使用相关的特征。



"""
