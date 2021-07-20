# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/20 14:26
# @Function: 随机森林：https://www.scikitlearn.com.cn/0.21.3/12/#1112
# 基于随机决策树的平均算法：
#   RandomForest 算法和 Extra-Trees 算法
#   专门为树而设计的扰动和组合技术
#   通过在分类器构造过程中引入随机性来创建一组不同的分类器
#   集成分类器的预测结果就是单个分类器预测结果的平均值

# 在随机森林中，集成模型中的每棵树构建时的样本都是由训练集经过有放回抽样得来的
# 在构建树的过程中进行结点分割时，选择的分割点是所有特征的最佳分割点，或特征的大小为 max_features 的随机子集的最佳分割点。
# 两种随机性的目的是降低估计器的方差


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier

"""
在iris数据集上绘制树集合的决策面
该图比较了由
决策树分类器（第一列）、随机林分类器（第二列）、额外树分类器（第三列）和AdaBoost分类器（第四列）学习的决策曲面。
在第一行，只使用萼片宽度和萼片长度特征构建分类器，在第二行，只使用花瓣长度和萼片长度，在第三行，只使用花瓣宽度和花瓣长度。


增加AdaBoost的最大深度会降低分数的标准差（但平均分数不会提高）。
"""
# Parameters
n_classes = 3
n_estimators = 30
cmap = plt.cm.RdYlBu
plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.5  # step widths for coarse classifier guesses
RANDOM_SEED = 13  # fix the seed on each iteration

iris = load_iris()

plot_idx = 1

models = [
    # max_depth和n_estimators只可做调整，对比查看
    DecisionTreeClassifier(max_depth=None),
    RandomForestClassifier(n_estimators=n_estimators),
    ExtraTreesClassifier(n_estimators=n_estimators),
    AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=n_estimators)
]

for pair in ([0, 1], [0, 2], [2, 3]):  # 只选两个相应的特征
    for model in models:  # 迭代模型
        X = iris.data[:, pair]
        y = iris.target

        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)  # 打乱数据
        X = X[idx]
        y = y[idx]

        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        # Train
        model.fit(X, y)

        scores = model.score(X, y)
        # 标题
        model_title = str(type(model)).split(".")[-1][:-2][:-len("Classifier")]

        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(len(model.estimators_))
        print(model_details + " with features", pair, "has a score of", scores)

        plt.subplot(3, 4, plot_idx)
        if plot_idx <= len(models):
            plt.title(model_title, fontsize=9)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

        # 绘制单个DecisionTreeClassifier或alpha混合分类器集合的决策面
        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            # 根据正在使用的估计器的数量选择alpha混合级别
            # （注意，如果AdaBoost在早期达到足够好的拟合，那么它可以使用比其最大值更少的估计器）
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # 构建一个更粗的网格来绘制一组集合分类，以显示这些分类与我们在决策曲面中看到的分类有何不同。
        # 这些点是规则的空间，没有黑色的轮廓
        xx_coarser, yy_coarser = np.meshgrid(
            np.arange(x_min, x_max, plot_step_coarser),
            np.arange(y_min, y_max, plot_step_coarser)
        )
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
        cs_points = plt.scatter(xx_coarser, yy_coarser, s=15, c=Z_points_coarser, cmap=cmap, edgecolors="none")

        # 绘制训练点，这些点聚集在一起，有一个黑色的轮廓
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['r', 'y', 'b']), edgecolor='k', s=20)
        plot_idx += 1  #

plt.suptitle("Classifiers on feature subsets of the Iris dataset", fontsize=12)
plt.axis("tight")
plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
plt.show()

"""
使用者这些方法最主要的是调整 n_estimators 和 max_features 参数

n_estimators是森林里树的数量，数量越多效果越好，随之计算时间也增加
当树的数量超过一个临界值之后，算法的效果并不会很显著地变好

max_features是分割节点时考虑的特征的随机子集的大小。
这个值越低，方差减小得越多，但是偏差的增大也越多。
max_features = None 总是考虑所有的特征
分类问题使用 max_features = "sqrt"是比较好的默认值 （随机考虑 sqrt(n_features) 特征，其中 n_features 是特征的个数）

"""

