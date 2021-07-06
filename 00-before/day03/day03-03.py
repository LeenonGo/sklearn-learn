"""
学习和预测
https://www.scikitlearn.com.cn/0.21.3/51/#_3
"""
from sklearn import svm
from sklearn import datasets
clf = svm.SVC(gamma=0.001, C=100.)

"""
clf：估计器： 预测 未知的样本所属的类。分类器。  黑箱
gamma ： 模型的参数，可通过 网格搜索 及 交叉验证 等工具，自动找到参数的良好值。
模型中 learn（学习）：将我们的训练集传递给 fit 方法来完成
"""
digits = datasets.load_digits()
clf.fit(digits.data[:-1], digits.target[:-1])  # fit学习
out = clf.predict(digits.data[-1:])  # predict预测

