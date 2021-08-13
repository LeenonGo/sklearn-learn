# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/13 16:48
# @Function: Pipeline: 链式评估器 https://www.scikitlearn.com.cn/0.21.3/38/#511-pipeline
# 
# Pipeline 可以把多个评估器链接成一个.这个是很有用的，因为处理数据的步骤一般都是固定的，例如特征选择、标准化和分类。
# 用途：
#   便捷性和封装性 只要对数据调用 fit和 predict 一次来适配所有的一系列评估器。
#   联合的参数选择 可以一次grid search管道中所有评估器的参数。
#   安全性 训练转换器和预测器使用的是相同样本，管道有助于防止来自测试数据的统计数据泄露到交叉验证的训练模型中。
#
# 1. 用法
# 1.1 构造 Pipeline 使用一系列 (key, value) 键值对来构建,其中 key 是你给这个步骤起的名字， value 是一个评估器对象:

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

estimators = [('reduce_dim', PCA()), ('clf', SVC())]
pipe = Pipeline(estimators)
print(pipe)
# 功能函数 make_pipeline 是构建管道的缩写; 它接收多个评估器并返回一个管道，自动填充评估器名:
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Binarizer

p = make_pipeline(Binarizer(), MultinomialNB())
print(p)

# 1.2
# 管道中的评估器作为一个列表保存在 steps 属性内,但可以通过索引或名称([idx])访问管道:
print("----------------------------------------------")
print(pipe.steps[0])
print(pipe[0])
print(pipe['reduce_dim'])
# 管道的named_steps属性允许在交互式环境中使用tab补全,以按名称访问步骤:
print(pipe.named_steps.reduce_dim is pipe['reduce_dim'])

# 1.3 嵌套参数
# 管道中的评估器参数可以通过 <estimator>__<parameter> 语义来访问:
print("----------------------------------------------")
print(pipe.set_params(clf__C=10))
# 对网格搜索尤其重要
from sklearn.model_selection import GridSearchCV
param_grid = dict(reduce_dim__n_components=[2, 5, 10], clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=param_grid)
print(grid_search)
# 单独的步骤可以用多个参数替换，除了最后步骤，其他步骤都可以设置为 passthrough 来跳过
from sklearn.linear_model import LogisticRegression
param_grid = dict(reduce_dim=['passthrough', PCA(5), PCA(10)], clf=[SVC(), LogisticRegression()], clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=param_grid)
print(grid_search)

# 2. 注意：
# 对管道调用 fit 方法的效果跟依次对每个评估器调用 fit 方法一样, 都是transform 输入并传递给下个步骤。
# 管道中最后一个评估器的所有方法，管道都有。

# 3. 缓存转换器：避免重复计算
# 适配转换器是很耗费计算资源的。设置了memory 参数， Pipeline 将会在调用fit方法后缓存每个转换器。










