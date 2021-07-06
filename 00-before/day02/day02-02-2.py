"""
学习predict_proba、predict、decision_function：展示模型对于输入样本的评判结果

"""
import numpy as np
from sklearn.svm import SVC
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# classes_属性:
# 在sklearn中，对于训练好的分类模型，模型都有一个classes_属性，classes_属性中按顺序保存着训练样本的类别标记
# 该输出结果的顺序就对应后续要说predict_proba、predict、decision_function输出结果的顺序或顺序组合

# 1. 样本标签从0开始的场景下训练分类模型
# x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1], [-1, 1], [-1, 2], [1, -1], [1, -2]])
# y = np.array([2, 2, 3, 3, 0, 0, 1, 1])
# clf = LogisticRegression()
# clf.fit(x, y)
# print(clf.classes_)  # [0 1 2 3]

# # 2. 样本标签不是从0开始的场景下训练分类模型
# x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1], [-1, 1], [-1, 2], [1, -1], [1, -2]])
# y = np.array([6, 6, 2, 2, 4, 4, 8, 8])
# clf2 = LogisticRegression()
# clf2.fit(x, y)
# print(clf2.classes_)  # [2 4 6 8]
#
# #
# print(clf2.predict_proba([[-1, -1]]))  # [[0.12532009 0.63284608 0.20186022 0.03997361]]
# # 说明：
# # 这行代码输出的就是对于clf2预测[[-1, -1]]类别的值
# # 输出的结果为[[0.12532009 0.63284608 0.20186022 0.03997361]]
# # 在训练数据中[-1, -1]属于类别6，
# # 在predict_proba输出概率中，最大概率值出现在第三个位置上，第三个位置对应的classes_类别刚好也是类别6 。
# # 也就是说，predict_proba输出概率最大值索引位置对应的classes_元素就是样本所属的类别


# print(clf2.predict([[-1, -1]]))  # [6]


#
x = np.array([[1, 2, 3], [1, 3, 4], [2, 1, 2], [4, 5, 6], [3, 5, 3], [1, 7, 2]])
y = np.array([3, 3, 3, 2, 2, 2])

clf = SVC(probability=True)
clf.fit(x, y)
print(clf.decision_function(x))

# 返回array([2, 3])，其中2为negetive，3为positive
print(clf.classes_)
# decision_function的函数说明：
# Returns
# -------
# X : ndarray of shape (n_samples, n_classes * (n_classes-1) / 2)
# Returns the decision function of the sample for each class
# in the model.
# If decision_function_shape='ovr', the shape is (n_samples,
# n_classes).
# 返回值：返回一个 (n_samples, n_classes * (n_classes-1) / 2)的数组。  参数decision_function_shape="ovo"
# 为模型中的每个类返回样本的决策函数。
# 如果参数decision_function_shape='ovr'，返回的形状是(n_samples, n_classes)

# 先解释一下什么是ovo he ovr
#
# ovr: "one vs rest" ,
# 假如训练数据中包含[0, 1, 2, 3]四个分类，那么分别将
# (1)  0 作为正样本，其余的1, 2, 3作为负样本;
# (2)  1 作为正样本，其余的0, 2, 3作为负样本;
# (3)  2 作为正样本，其余的0, 1, 2作为负样本;
# (4)  3 作为正样本，其余的0, 1, 2作为负样本;
# 训练4个分类器，每个分类器预测的结果表示属于对应正类也就是0， 1， 2， 3 的概率。
# 这样对于一个输入样本就相当于要进行4个二分类，然后取输出结果最大的数值对应的classes_类别。

# ovo: "One-vs-One"。车轮战术。
# 同样，假如训练数据中包含[0, 1, 2, 3]四个分类，
# 先将类别0作为正样本，类别1，类别2，类别3依次作为负样本训练3个分类器，
# 然后以类别1为正样本，类别0，类别2， 类别3作为负样本训练3个分类器，以此类推。
# 由于类别0为正样本，类别1为负样本和类别1为正样本、类别0为负样本实质上是一样的，所以不需要重复训练。

# 综上。训练样本有n_classes个类别，则
# 'ovr'模式需要训练n_classes个分类器，
# ‘ovo’模式需要训练n_classes * (n_classes - 1) / 2 个分类器

# 对于SVM来说，有多少个分类器就得有多少个分隔超平面，有多少个分隔超平面就得有多少个decision_function值

# 二分类模型中，decision_function返回的数组形状等于样本个数，
# 也就是一个样本返回一个decision_function值.
# 并且，此时的decision_function_shape参数失效 ，因为只需要训练一个分类器。(无关"ovr"与"ovo")
# classes_中的第一个标签代表是负样本，第二个标签代表正样本。 [2 3]中, 2为negetive，3为positive
# 如上面代码输出的值  [ 1.00089036  0.64493601  0.97960658  -1.00023781 -0.9995244  -1.00023779]

# decision_function的符号
# 大于0表示正样本的可信度大于负样本，否则可信度小于负样本。 选择可信度高的作为预测值，
# 即：表示前3个样本为类别3，后3个样本为类别2

print("-" * 50)

# 对于多分类的decision_function
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1], [-1, 1], [-1, 2], [1, -1], [1, -2]])
y = np.array([2, 2, 3, 3, 0, 0, 1, 1])  # 可以得到的信息：（1） classes_: [0 1 2 3] （2） ovr:4  ovo: 6
clf = SVC(probability=True, decision_function_shape="ovr")  # SVC多分类模型默认采用ovr模式
clf.fit(X, y)

# One-vs-One 按照decision_function的得分[01, 02, 03, 12, 13, 23]判断每个分类器的分类结果，然后进行投票
# One-vs-Rest 选择decision_function的得分[0-Rest, 1-Rest, 2-Rest, 3-Rest]最大的作为分类结果

print("decision_function:\n", clf.decision_function([[-1, -1]]))
print("predict:\n", clf.predict([[-1, -1]]))  # precidt预测样本对应的标签类别
print("predict_proba:\n", clf.predict_proba([[-1, -1]]))  # predict_proba 预测样本对应各个类别的概率 这个是得分,每个分类器的得分，取最大得分对应的类。

# decision_function输出的最大值对应的正样本类别就是decision_function认为置信度最高的预测类别
print("-" * 50)
print("-" * 50)
print("-" * 50)

# ovo场景：
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1], [-1, 1], [-1, 2], [1, -1], [1, -2]])
y = np.array([2, 2, 3, 3, 0, 0, 1, 1])
clf = SVC(probability=True, decision_function_shape="ovo")
clf.fit(X, y)

print("decision_function:\n", clf.decision_function([[-1, -1]]))
# 输出的结果为： [[-0.07609727 -1.00023294  0.27849207 -0.83425862  0.24756982  1.00006256]]
# 分析：
# -0.07609727对应01分类器，且数值小于0，则分类结果为后者，即类别1
# -1.00023294对应02分类器，且数值小于0，则分类结果为后者，即类别2
# 0.27849207对应03分类器，且数值大于0，则分类结果为前者，即类别0
# -0.834258626对应12分类器，且数值小于0，则分类结果为后者，即类别2
# 0.24756982对应13分类器，且数值大于0，则分类结果为前者，即类别1
# 1.00006256对应23分类器，且数值大于0，则分类结果为前者，即类别2
# 最终得票数：{类别0: 1， 类别1: 2, 类别2: 3， 类别3: 0}
# 对以上分类结果voting投票，多数获胜，即最终分类结果为类别2。


"""
1. predict_proba： 输出样本属于各个类别的概率，取概率最大的类别作为样本的预测结果
2. predict： 预测输入样本所属的类别
3. decision_function： 计算样本距离每个分类边界的距离，并由此可以推算出predict的预测结果
"""
