# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/23 10:14
# @Function: 基于MNIST的MLP权重可视化
# 有时观察神经网络的学习系数可以提供对学习行为的洞察。
# 例如，如果权重看起来是非结构化的，可能有些根本没有使用，或者如果存在非常大的系数，可能正则化太低或者学习率太高。
# 这个例子展示了如何在MNIST数据集上训练的MLPClassifier中绘制一些第一层权重。
# 输入数据由28x28像素的手写数字组成，数据集中有784个特征。因此，第一层权重矩阵具有形状（784，隐藏层大小[0]）。
# 因此，我们可以将权重矩阵的一列可视化为28x28像素的图像。
# 为了使示例运行得更快，我们使用很少的隐藏单元，并且只训练很短的时间。
# 训练时间越长，负重的空间外观就越平滑。
# 这个例子会抛出一个警告，因为它不收敛，在这个例子中，这是我们想要的，因为CI的时间限制。


import warnings
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

print(__doc__)

# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.

# rescale the data, use the traditional train/test split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)

# this example won't converge because of CI's time constraints, so we catch the
# warning and are ignore it here
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin, vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()

