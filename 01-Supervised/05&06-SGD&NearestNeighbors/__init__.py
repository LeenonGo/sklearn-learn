# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/12 10:22
# @Function: 1.SGD 随机梯度下降  2. NearestNeighbors 最近邻

# #####################1.SGD 随机梯度下降#################################
# 主要用于凸损失函数下线性分类器的判别式学习。如SVM和Logistic Regression
#
"""
优势：
    高效
    易于实现
劣势：
    需要设置超参数，如正则化
    对特征缩放敏感，因此需要对数据进行缩放。


"""
"""
关于数据的缩放：
    如输入向量 X 上的每个特征缩放到 [0,1] 或 [- 1，+1]
    或将其标准化，使其均值为 0，方差为 1。
如果属性有固定的尺度，比如词频，就不必缩放。 最好使用 GridSearchCV 找到一个合理的正则化项
"""

# 使用StandardScaler缩放数据：
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = []
X_test = []
scaler.fit(X_train)  # Don’t cheat - fit only on training data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # apply same transformation to test data

# #####################2. NearestNeighbors 最近邻#################################
"""
最近邻方法的原理：
    从训练样本中找到与新点在距离上最近的预定数量的几个点，然后从这些点中预测标签。
    这些点的数量由用户决定（K-最近邻学习），也可以根据不同的点的局部密度（基于半径的最近邻学习）确定
"""



