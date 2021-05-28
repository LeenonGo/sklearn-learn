# 图像显示的是logistic回归分类器在鸢尾花数据集的前两个维度（萼片长度和宽度）上决定边界。数据点根据其标签上色。
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

# Create an instance of Logistic Regression Classifier and fit the data.
logreg = LogisticRegression(C=1e5)
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
print("x_min, x_max:  ", x_min, x_max)
print("y_min, y_max:  ", y_min, y_max)
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))   # 将图形分为网格  输出坐标矩阵
# meshgrid这一步花了一点时间理解，x和y（第一参数和第二参数）都是数组，可以将这里理解为在坐标轴上讲这些值画出来，然后分别垂直于y或x轴画垂线，得到的网格
# 输出的值就是这些垂线的交点对应轴线的坐标
# 这样理解：比如交点坐标为
# [
# [(0,1), (1,1), (2,1)],
# [(0,0), (1,0), (2,0)]
# ]
# 那么xx 就是
# [
# [ 0 1 2 ]
# [ 0 1 2 ]
# ]
# yy 就是
# [
# [ 1 1 1 ]
# [ 0 0 0 ]
# ]

zz = np.c_[xx.ravel(), yy.ravel()]  # 将xx和yy按行连接 得到的就是交点的坐标
Z = logreg.predict(zz)  # 对坐标点进行预测  logreg拟合时（第14行）对应的参数其实也是坐标，与zz是对应的

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)   # 第一参数对应于x坐标，第二参数对应于y坐标，第三参数对应于每个坐标点（x, y）的分类结果。

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())

plt.show()

"""
思考：
在最开始拟合时， 分类器选择的参数X和Y是数据集前两个特征和目标
先理解为：萼片长度和宽度对鸢尾花类别的影响，
即再次理解为： 分类器的输入为萼片长度和宽度， 为了得到鸢尾花的的类别
（其实有的说法不是很准确，但是先这样理解）


然后，对萼片长度和宽度按0.02的步长从min到max离散成多个点（其实可以看作连续），得到这些点的连接值
这里理解为一个坐标
第66行的理解将输入的长度和宽度作为x轴和y轴取值，这里离散了多个值之后仍然对应长度和宽度。
前面输入长度和宽度做训练，这里也输入长度和宽度做预测（第43行）
得到的输出就是对鸢尾花类别的预测

从图像上看，因为鸢尾花的类别有3种，因此图像上按颜色分成了3个区域，这是按预测值分的；
散点图上的点是数据集中的数据绘制的，可以对比得出预测结果的准确性如何。


"""







