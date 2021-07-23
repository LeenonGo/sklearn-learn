# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/23 9:20
# @Function: https://www.scikitlearn.com.cn/0.21.3/18/#1172
# 
from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)

print(clf.predict([[2., 2.], [-1., -2.]]))
print([coef.shape for coef in clf.coefs_])
# clf.coefs_ 包含了构建模型的权值矩阵

# 目前， MLPClassifier 只支持交叉熵损失函数，通过运行 predict_proba 方法进行概率估计。
# 使用了通过反向传播计算得到的梯度和某种形式的梯度下降来进行训练
# 最小化交叉熵损失函数，为每个样本 x 给出一个向量形式的概率估计 P(y|x)
print(clf.predict_proba([[2., 2.], [1., 2.]]))
# [[1.96718015e-04 9.99803282e-01]
#  [1.96718015e-04 9.99803282e-01]]
# 表示预测[2., 2.]为标签0的概率为1.96718015e-04， 为标签1的概率为9.99803282e-01


# 此外，该模型支持 多标签分类 ，一个样本可能属于多个类别。
# 对于每个类，原始输出经过 logistic 函数变换后，大于或等于 0.5 的值将进为 1，否则为 0。
# 对于样本的预测输出，值为 1 的索引位置表示该样本的分类类别
X = [[0., 0.], [1., 1.]]
y = [[0, 1], [1, 1]]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf.fit(X, y)
print(clf.predict([[1., 2.]]))
print(clf.predict([[0., 0.]]))


