from sklearn import linear_model
from sklearn import datasets
import numpy as np
alphas = np.logspace(-4, -1, 6)

diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

regr = linear_model.Lasso()
scores = [regr.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train).score(diabetes_X_test, diabetes_y_test)for alpha in alphas]
best_alpha = alphas[scores.index(max(scores))]  # 取出最高分
regr.alpha = best_alpha
regr.fit(diabetes_X_train, diabetes_y_train)  # 重新拟合
print(regr.coef_)  # 得出参数

"""
Lasso 回归模型

scikit-learn 里 Lasso 对象使用 coordinate descent（坐标下降法） 方法解决 lasso 回归问题，对于大型数据集很有效。
"""

