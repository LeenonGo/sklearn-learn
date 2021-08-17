# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/17 13:12
# @Function:5.9. 预测目标 (y) 的转换  https://www.scikitlearn.com.cn/0.21.3/46/

# 本章要介绍的这些变换器不是被用于特征的，而是只被用于变换监督学习的目标。

# 5.9.1. 标签二值化
# LabelBinarizer 是一个用来从多类别列表创建标签矩阵的工具类:
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit([1, 2, 6, 4, 2])
print(lb.classes_)
print(lb.transform([1, 6]))
# 对于多类别是实例，可以使用 MultiLabelBinarizer:
lb = preprocessing.MultiLabelBinarizer()
print(lb.fit_transform([(1, 2), (3,)]))
print(lb.classes_)
print("-------------------------------------------")

# 5.9.2. 标签编码
# LabelEncoder 是一个可以用来将标签规范化的工具类，它可以将标签的编码值范围限定在[0,n_classes-1].
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit([1, 2, 2, 6])
print(le.transform([1, 1, 2, 6]))
print(le.inverse_transform([0, 0, 1, 2]))
# 也可以用于非数值型标签的编码转换成数值标签（只要它们是可哈希并且可比较的）:
le.fit(["paris", "paris", "tokyo", "amsterdam"])
print(list(le.classes_))
print(le.transform(["tokyo", "tokyo", "paris"]))
print(list(le.inverse_transform([2, 2, 1])))
