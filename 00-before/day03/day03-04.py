from sklearn import svm
from sklearn import datasets

# 内置持久化模块
import pickle
from joblib import dump, load


clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)
s = pickle.dumps(clf)  # 保存模型
clf2 = pickle.loads(s)
clf2.predict(X[0:1])

dump(clf, 'filename.joblib')  # 对大数据更有效，但只能序列化 (pickle) 到磁盘而不是字符串变量
clf = load('filename.joblib')  # 加载模型

