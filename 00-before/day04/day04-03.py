# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/6/3 17:56
# @Function: 在数字数据集中，用一个线性内核绘制一个 SVC 估计器的交叉验证分数来作为 C 参数函数(使用从1到10的点对数网格).

# 在数字数据集上使用支持向量机交叉验证的教程练习。
# 本练习用于科学数据处理统计学习教程的“模型选择：选择估计量及其参数”部分的交叉验证生成器部分。
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm
import matplotlib.pyplot as plt

X, y = datasets.load_digits(return_X_y=True)

svc = svm.SVC(kernel='linear')
C_s = np.logspace(-10, 0, 10)

scores = list()
scores_std = list()
for C in C_s:
    svc.C = C
    this_scores = cross_val_score(svc, X, y, n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

# Do the plotting
plt.figure()
plt.semilogx(C_s, scores)
plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
locs, labels = plt.yticks()
plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
plt.ylabel('CV score')
plt.xlabel('Parameter C')
plt.ylim(0, 1.1)
plt.show()











