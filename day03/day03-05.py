"""
scikit-learn分类器的规定：

1. 除非特别指定，输入将被转换为 float64
2. 回归目标被转换为 float64 ，但分类目标维持不变
    clf.fit(iris.data, iris.target_names[iris.target])
    clf.predict(iris.data[:3]) 返回一个字符串数组
3. 估计器的超参数可以通过set_params 方法在实例化之后进行更新。
    调用 fit() 多次将覆盖以前的 fit() 所学到的参数
4. 当使用 多类分类器 时，执行的学习和预测任务取决于参与训练的目标数据的格式

"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# 4
X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

classif = OneVsRestClassifier(estimator=SVC(random_state=0))
print(classif.fit(X, y).predict(X))


print("----------------------------------")
y = LabelBinarizer().fit_transform(y)  # 将目标向量 y 转化成二值化后的二维数组
# print(y)
print(classif.fit(X, y).predict(X))  # 返回的全零向量表示不能匹配用来训练中的目标标签中的任意一个


print("----------------------------------")
# 使用多标签输出，类似地可以为一个实例分配多个标签
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y = MultiLabelBinarizer().fit_transform(y)
print(classif.fit(X, y).predict(X))



