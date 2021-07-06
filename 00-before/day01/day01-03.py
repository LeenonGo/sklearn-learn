from __future__ import print_function
from sklearn import datasets
from sklearn import linear_model
import numpy as np

"""
线性模型：从回归到稀疏
糖尿病数据集:糖尿病数据集包括442名患者的10个生理特征，和一年后的疾病级别指标
(年龄， age
性别， sex
体质指数，bmi
血压，bp
六项血清测量值（总胆固醇tc、低密度脂蛋白ldl、高密度脂蛋白hdl、总胆固醇/高密度脂蛋白tch、可能是血清甘油三酯水平的对数ltg、血糖水平glu）)
"""
diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# 任务是使用生理特征来预测疾病级别。
# 线性回归最简单的拟合线性模型形式，是通过调整数据集的一系列参数令残差平方和尽可能小
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
# print(regr.coef_)  # 参数/权重值？
predict_X_test = regr.predict(diabetes_X_test)
dx = np.mean((predict_X_test-diabetes_y_test)**2)
# print(dx)  # 方差  衡量数据的波动性

# 方差分数：1 是完美的预测;0 意味着 X 和 y 之间没有线性关系。
score = regr.score(diabetes_X_test, diabetes_y_test)
print(score)

# 如果每个维度的数据点很少，观察噪声就会导致很大的方差
# day01-03-2.py
# day01-03-2a.py

regr = linear_model.Ridge(alpha=.1)
alphas = np.logspace(-4, -1, 6)
print([
    regr.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train,).score(diabetes_X_test, diabetes_y_test) for alpha in alphas]
)






