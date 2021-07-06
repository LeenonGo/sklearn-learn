import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

"""
监督学习：从高维观察预测输出变量

监督学习：在于学习两个数据集之间的联系
"""

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

# import some data to play with
iris = datasets.load_iris()
"""
print(iris.DESCR)  # 查看数据集详细信息
150个样本，每一类50个
4种属性，预测值，分类
属性信息：  萼片sepal长度  萼片宽度  花瓣petal长度  花瓣宽度
类   别：   山鸢尾（Setosa）   变色（杂色）鸢尾Versicolour  维吉尼亚鸢尾Virginica
"""
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target  # 花的种类
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5  # 第一个属性最小值最大值
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5  # 第二个属性最小值最大值

"""绘制散点图"""
# Plot the training points

plt.figure(2, figsize=(8, 6))  # 图像编号为2 宽高为8x6 (英寸)
plt.clf()  # 界面清理

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')  # 散点图 x, y c=color edgecolor散点的边缘颜色
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())  # 显示x轴的刻标以及对应的标签 pltxticks( arange(5), ('Tom', 'Dick', 'Harry', 'Sally', 'Sue') )
plt.yticks(())


"""绘制3维图形"""
# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)  # PCA(Principal Component Analysis) 降维
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
#
plt.show()
