import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)

# K近邻分类器
# 将鸢尾属植物数据集分解为训练集和测试集
# 随机排列，用于使分解的数据随机分布
np.random.seed(0)  # 设置相同的seed，每次生成的随机数相同
indices = np.random.permutation(len(iris_X))  # 随机排序
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

# 创建和拟合一个最近邻分类器
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
print(knn.predict(iris_X_test))
print(iris_y_test)



