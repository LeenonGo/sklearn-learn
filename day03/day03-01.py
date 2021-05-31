"""
今日学习  使用 scikit-learn 介绍机器学习
https://www.scikitlearn.com.cn/0.21.3/51/#_1

对scikit-learn 过程中用到的 机器学习 词汇进行用例子进行阐述


机器学习：
    1. 监督学习：数据中带有一个想要预测的值，可分为：
        分类：样本属于两个或更多个类的情况，
        回归：期望的输出由一个或多个连续变量组成
    2. 无监督学习：数据中没有目标值，聚类或密度估计

训练集：从中学习数据的属性
测试集：测试性质

"""

from sklearn import datasets

# 加载数据集，数据集是(n_samples, n_features) 数组
iris = datasets.load_iris()
digits = datasets.load_digits()

data = digits.data  # 得到用于分类的样本特征
target = digits.target  # 数据集内每个数字的真实类别
