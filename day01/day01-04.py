"""稀疏"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets, linear_model

X, y = datasets.load_diabetes(return_X_y=True)
indices = (0, 1)

X_train = X[:-20, indices]
X_test = X[-20:, indices]
y_train = y[:-20]
y_test = y[-20:]

ols = linear_model.LinearRegression()
ols.fit(X_train, y_train)


# #############################################################################
# Plot the figure
def plot_figs(fig_num, elev, azim, X_train, clf):
    fig = plt.figure(fig_num, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, elev=elev, azim=azim)  # elev和azim表示转动坐标轴

    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='k', marker='+')  # 前两个特征与目标的三维图
    ax.plot_surface(np.array([[-.1, -.1], [.15, .15]]),
                    np.array([[-.1, .15], [-.1, .15]]),
                    clf.predict(np.array([[-.1, -.1, .15, .15],
                                          [-.1, .15, -.1, .15]]).T
                                ).reshape((2, 2)),
                    alpha=.5)
    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_zlabel('Y')
    # ax.w_xaxis.set_ticklabels([])
    # ax.w_yaxis.set_ticklabels([])
    # ax.w_zaxis.set_ticklabels([])


# Generate the three different figures from different views
# elev = 43.5
# azim = -110
# plot_figs(1, elev, azim, X_train, ols)
#
# elev = -.5
# azim = 0
# plot_figs(2, elev, azim, X_train, ols)

elev = -.5
azim = 90
plot_figs(3, elev, azim, X_train, ols)

plt.show()

"""
图像显示糖尿病数据集的特征1和特征2。
它说明，尽管特征2在整个模型上有很强的系数，但与特征1相比，它并没有给我们太多关于y的信息

y表示一年后的疾病级别指标

特征2为性别属性
单独看x2和y的关系  可以看出男性和女性在一年后的指标的分布

特征1为年龄属性
单独看x1和y的关系  可以看出年龄与指标的分布


三维立体的看得出结论：在相同的年龄下 指标的分布与性别关联性不大

"""






