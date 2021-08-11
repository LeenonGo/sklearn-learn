# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/11 21:40
# @Function: 3.5.1. 验证曲线
# 
# 我们需要一个评分函数来验证一个模型， 例如分类器的准确性。
# 在此图中，您可以看到不同核参数gamma值的SVM的训练分数和验证分数。对于非常低的gamma值，您可以看到训练分数和验证分数都很低。这被称为欠拟合。
# gamma的中间值将导致两个分数的高值，即分类器表现相当好。如果gamma太高，分类器将过度拟合，这意味着训练分数很好，但验证分数很差。
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

X, y = load_digits(return_X_y=True)

param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(
    SVC(), X, y, param_name="gamma", param_range=param_range, scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SVM")
plt.xlabel(r"$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)
plt.legend(loc="best")
plt.show()

