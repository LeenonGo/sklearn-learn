# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/12 16:05
# @Function: 4.1. 部分依赖图 PDP  https://www.scikitlearn.com.cn/0.21.3/36/#41-%E9%83%A8%E5%88%86%E4%BE%9D%E8%B5%96%E5%9B%BE
# 显示了目标响应和一组“目标”特征之间的依赖关系,并边缘化所有其他特征（“补充”特征）的值。
# 由于人类感知的限制，目标特征集的大小必须很小（通常是一个或两个），因此目标特征通常需要从最重要的特征中选择。


# 示例：部分依赖与个体条件期望图
# 部分依赖图显示了目标函数和一组感兴趣的特征之间的依赖关系，边缘化了所有其他特征（补充特征）的值。

# 类似地，单个条件期望（ICE）图显示了目标函数和感兴趣的特征之间的依赖关系。
# 然而，与显示感兴趣特征的平均效果的部分依赖图不同，ICE图分别显示了预测对每个样本特征的依赖性，每个样本一行。
# ICE图只支持一个感兴趣的功能。

# 此示例显示如何从加利福尼亚住房数据集上训练的MLPrepressor和HistGradientBoostingRegression中获取部分相关性和ICE图。


# 加州住房数据预处理
# 中心目标化以避免梯度提升初始偏差：使用“递归”方法的梯度提升不考虑初始估值器（此处默认为平均目标）。

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

cal_housing = fetch_california_housing()
X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
y = cal_housing.target

y -= y.mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# 不同模型的单向部分相关
# 在本节中，我们将使用两种不同的机器学习模型计算单向部分依赖：（i）多层感知器和（ii）梯度提升。
# 通过这两个模型，我们说明了如何计算和解释部分依赖图（PDP）和个体条件期望（ICE）。

# Multi-layer perceptron
from time import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.neural_network import MLPRegressor

print("Training MLPRegressor...")
tic = time()
est = make_pipeline(QuantileTransformer(),  # 利用分位数信息变换特征
                    MLPRegressor(hidden_layer_sizes=(50, 50), learning_rate_init=0.01, early_stopping=True)
                    )
est.fit(X_train, y_train)
print(f"done in {time() - tic:.3f}s")
print(f"Test R2 score: {est.score(X_test, y_test):.2f}")
# 我们配置了一个管道来缩放数字输入特征，并调整了神经网络的大小和学习速率，以便在测试集上在训练时间和预测性能之间取得合理的折衷。
# 重要的是，这个表格数据集的特性具有非常不同的动态范围。神经网络往往对不同尺度的特征非常敏感，忘记对数值特征进行预处理会导致模型非常差。
# 使用更大的神经网络可以获得更高的预测性能，但训练成本也会显著增加。

# 请注意，在绘制部分相关性之前，检查模型在测试集上是否足够准确是很重要的，因为在解释给定特征对不良模型预测功能的影响时几乎没有用处。
# 我们将绘制部分相关性，包括单个相关性（ICE）和平均相关性（PDP）。我们仅限制ICE条冰曲线，以避免地块过度拥挤。

import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence

print('Computing partial dependence plots...')
tic = time()
features = ['MedInc', 'AveOccup', 'HouseAge', 'AveRooms']
display = plot_partial_dependence(
    est, X_train, features, kind="both", subsample=50, n_jobs=3, grid_resolution=20, random_state=0
)
print(f"done in {time() - tic:.3f}s")
display.figure_.suptitle(
    'Partial dependence of house value on non-location features\n'
    'for the California housing dataset, with MLPRegressor'
)
display.figure_.subplots_adjust(hspace=0.3)


# Gradient boosting
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor

print("Training HistGradientBoostingRegressor...")
tic = time()
est = HistGradientBoostingRegressor()
est.fit(X_train, y_train)
print(f"done in {time() - tic:.3f}s")
print(f"Test R2 score: {est.score(X_test, y_test):.2f}")

print('Computing partial dependence plots...')
tic = time()
display = plot_partial_dependence(
    est, X_train, features, kind="both", subsample=50,
    n_jobs=3, grid_resolution=20, random_state=0
)
print(f"done in {time() - tic:.3f}s")
display.figure_.suptitle(
    'Partial dependence of house value on non-location features\n'
    'for the California housing dataset, with Gradient Boosting'
)
display.figure_.subplots_adjust(wspace=0.4, hspace=0.3)

plt.show()








