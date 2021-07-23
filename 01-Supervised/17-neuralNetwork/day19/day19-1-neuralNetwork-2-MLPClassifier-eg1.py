# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/23 9:43
# @Function: 比较MLP分类器的随机学习策略
# 
"""
这个例子展示了不同随机学习策略的一些训练损失曲线，包括SGD和Adam。
由于时间限制，我们使用了几个小数据集，其中L-BFGS可能更适合。
然而，这些示例中显示的总体趋势似乎会延续到更大的数据集。
请注意，这些结果可能高度依赖于learning_rate_init的值。
"""

import warnings
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning

# different learning rate schedules and momentum parameters
# learning_rate_init
params = [
    {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0, 'learning_rate_init': 0.2},
    {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9, 'learning_rate_init': 0.2,
     'nesterovs_momentum': False},
    {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9, 'learning_rate_init': 0.2,
     'nesterovs_momentum': True},
    {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0, 'learning_rate_init': 0.2},
    {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9, 'learning_rate_init': 0.2,
     'nesterovs_momentum': True},
    {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9, 'learning_rate_init': 0.2,
     'nesterovs_momentum': False},
    {'solver': 'adam', 'learning_rate_init': 0.01}
]

labels = ["constant learning-rate",
          "constant with momentum",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate",
          "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum",
          "adam"]

plot_args = [
    {'c': 'red', 'linestyle': '-'},
    {'c': 'green', 'linestyle': '-'},
    {'c': 'blue', 'linestyle': '-'},
    {'c': 'red', 'linestyle': '--'},
    {'c': 'green', 'linestyle': '--'},
    {'c': 'blue', 'linestyle': '--'},
    {'c': 'black', 'linestyle': '-'}
]


def plot_on_dataset(X, y, ax, name):
    # 对于每个数据集，为每个学习策略绘制学习图
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)

    X = MinMaxScaler().fit_transform(X)
    mlps = []
    if name == "digits":
        # digits较大，但收敛速度相当快
        max_iter = 15
    else:
        max_iter = 400

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(random_state=0, max_iter=max_iter, **param)

        # 一些参数组合不会收敛，如图中所示，因此此处忽略它们
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
            mlp.fit(X, y)

        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
        ax.plot(mlp.loss_curve_, label=label, **args)


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# 加载/生成一些小型数据集
iris = datasets.load_iris()
X_digits, y_digits = datasets.load_digits(return_X_y=True)
data_sets = [(iris.data, iris.target),
             (X_digits, y_digits),
             datasets.make_circles(noise=0.2, factor=0.5, random_state=1),
             datasets.make_moons(noise=0.3, random_state=0)]

for ax, data, name in zip(axes.ravel(), data_sets, ['iris', 'digits', 'circles', 'moons']):
    plot_on_dataset(*data, ax=ax, name=name)
    """
    参数带*：
        参数前面加上* 号 ，意味着参数的个数不止一个
        带一个星号（*）参数的函数传入的参数存储为一个元组（tuple）
        带两个（*）号则是表示字典（dict）
        此外，一个（*）号还可以解压参数列表
    """

fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
plt.show()
