# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/6/3 17:41
# @Function: 交叉验证生成器

from sklearn.model_selection import KFold, cross_val_score
from sklearn import datasets, linear_model, svm

svc = svm.SVC(C=1, kernel='linear')

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

X = ["a", "a", "b", "c", "c", "c"]
k_fold = KFold(n_splits=3)
for train_indices, test_indices in k_fold.split(X):
    print('Train: %s | test: %s' % (train_indices, test_indices))

scores = [svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test]) for train, test in k_fold.split(X_digits)]
print(scores)

# 使用cross_val_score计算交叉验证分数
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y, cv=3))










