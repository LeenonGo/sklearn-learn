"""
分数和交叉验证分数
https://www.scikitlearn.com.cn/0.21.3/55/#_2
"""
from sklearn import datasets, svm
import numpy as np
# 每一个估计量都有一个可以在新数据上判定拟合质量(或预期值)的 score 方法。
# 这个分值越大越好.

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')
f = svc.fit(X_digits[:-100], y_digits[:-100])
score = f.score(X_digits[-100:], y_digits[-100:])
# print(score)

# 为了更好地预测精度(我们可以用它作为模型的拟合优度代理)，
# 我们可以连续分解用于我们训练和测试用的 折叠数据。
X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)
scores = list()
for k in range(3):
    # We use 'list' to copy, in order to 'pop' later on
    # 使用list拷贝，是为了后面的pop
    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
print(scores)
# 方法说明：
#   将数据分割成3份 做三次实验  每次实验用一份数据预测 剩下的两份数据训练
#   这被称为 KFold 交叉验证.
















