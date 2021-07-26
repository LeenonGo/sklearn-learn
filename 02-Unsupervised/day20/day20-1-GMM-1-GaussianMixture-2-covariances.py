# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/26 13:48
# @Function: 高斯混合  https://www.scikitlearn.com.cn/0.21.3/20/#211

# GaussianMixture 对象实现了用来拟合高斯混合模型的 期望最大化 (EM) 算法。
# 还可以为多变量模型绘制置信椭圆体，同时计算 BIC（Bayesian Information Criterion，贝叶斯信息准则）来评估数据中聚类的数量。
# 而其中的GaussianMixture.fit 方法可以从训练数据中拟合出一个高斯混合模型。
#

# 优点：
#   速度: 是混合模型学习算法中最快的算法．
#   无偏差性: 这个算法仅仅只是最大化可能性，并不会使均值偏向于0，或是使聚类大小偏向于可能适用或者可能不适用的特殊结构。
# 缺点:
#   奇异性:  当每个混合模型没有足够多的点时，会很难去估算对应的协方差矩阵，
#           同时该算法会发散并且去寻找具有无穷大似然函数值的解，除非人为地正则化相应的协方差。
#   分量的数量: 这个算法总是会使用它所能用的全部分量，所以在缺失外部线索的情况下，需要留存数据或者信息理论标准来决定用多少个分量。


# 示例：高斯混合模型几种协方差类型的证明。
# 虽然GMM通常用于聚类，但是我们可以将得到的聚类与数据集中的实际类进行比较。
# 这里用训练集中的类的平均值初始化高斯平均值，使这个比较有效。

# 使用鸢尾花数据集上的各种GMM协方差类型，在训练和保持测试数据上绘制预测标签。
# 例中比较了GMM中diagonal，spherical，tied，full协方差矩阵的性能。
# 尽管人们期望完全协方差在一般情况下表现最好，但它在小数据集上容易过度拟合，并且不能很好地推广到持久的测试数据。

# 在图上，训练数据显示为点，而测试数据显示为十字。iris数据集是四维的。这里只显示前两个维度，因此一些点在其他维度中是分开的。

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

colors = ['navy', 'turquoise', 'darkorange']


def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')


iris = datasets.load_iris()

# Break up the dataset into non-overlapping training (75%) and testing (25%) sets.
skf = StratifiedKFold(n_splits=4)
# Only take the first fold.
train_index, test_index = next(iter(skf.split(iris.data, iris.target)))

X_train = iris.data[train_index]
y_train = iris.target[train_index]
X_test = iris.data[test_index]
y_test = iris.target[test_index]

n_classes = len(np.unique(y_train))

# Try GMMs using different types of covariances.
estimators = {
    cov_type: GaussianMixture(n_components=n_classes, covariance_type=cov_type, max_iter=20, random_state=0)
    for cov_type in ['spherical', 'diag', 'tied', 'full']
}

n_estimators = len(estimators)

plt.figure(figsize=(3 * n_estimators // 2, 6))
plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05, left=.01, right=.99)

for index, (name, estimator) in enumerate(estimators.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    # 因为我们有训练数据的类标签，所以我们可以在有监督的方式下初始化GMM参数。
    estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                     for i in range(n_classes)])

    # Train the other parameters using the EM algorithm.
    estimator.fit(X_train)

    h = plt.subplot(2, n_estimators // 2, index + 1)
    make_ellipses(estimator, h)

    for n, color in enumerate(colors):
        data = iris.data[iris.target == n]
        plt.scatter(data[:, 0], data[:, 1], s=0.8, color=color, label=iris.target_names[n])
    # Plot the test data with crosses
    # for n, color in enumerate(colors):
        data2 = X_test[y_test == n]
        plt.scatter(data2[:, 0], data2[:, 1], marker='x', color=color)

    y_train_pred = estimator.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy, transform=h.transAxes)

    y_test_pred = estimator.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy, transform=h.transAxes)

    plt.xticks(())
    plt.yticks(())
    plt.title(name)

plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))

plt.show()
