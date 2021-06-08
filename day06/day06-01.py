# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/6/8 13:41
# @Function: 模型管道化 https://www.scikitlearn.com.cn/0.21.3/57/#_2
# 
"""
通过之前的学习我们已经知道：
一些模型可以做数据转换，一些模型可以用来预测变量。
我们可以建立一个组合模型同时完成以上工作

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
# 定义一个管道来搜索PCA截断和分类器正则化的最佳组合。
logistic = SGDClassifier(loss='log', penalty='l2', early_stopping=True,
                         max_iter=10000, tol=1e-5, random_state=0)  # 随机梯度下降，用于分类（预测）
pca = PCA()  # 主成分分析，用于降维
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    'pca__n_components': [5, 20, 30, 40, 50, 64],
    'logistic__alpha': np.logspace(-4, 4, 5),
}  # 给模型参数
# search = GridSearchCV(pipe, param_grid, iid=False, cv=5)  # iid参数被取消了
search = GridSearchCV(pipe, param_grid, cv=5)  # 网格搜索
search.fit(X_digits, y_digits)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

# Plot the PCA spectrum
pca.fit(X_digits)
logistic.fit(X_digits, y_digits)

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax0.plot(pca.explained_variance_ratio_, linewidth=2)  # 降维后的各主成分的方差值占总方差值的比例。值越大越重要。
ax0.set_ylabel('PCA explained variance')
ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.show()

