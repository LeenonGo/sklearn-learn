# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/13 17:23
# @Function: 5.1.3. FeatureUnion（特征联合）: 复合特征空间  https://www.scikitlearn.com.cn/0.21.3/38/#513-featureunion
#
# FeatureUnion 合并了多个转换器对象形成一个新的转换器，该转换器合并了他们的输出。一个 FeatureUnion 可以接收多个转换器对象
# 在适配期间，每个转换器都单独的和数据适配。 对于转换数据，转换器可以并发使用，且输出的样本向量被连接成更大的向量。
# FeatureUnion 功能与 Pipeline 一样- 便捷性和联合参数的估计和验证。
#
# 可以结合:FeatureUnion和 Pipeline 来创造出复杂模型。

# 用法：
#   一个 FeatureUnion 是通过一系列 (key, value) 键值对来构建的,
#   其中的 key 给转换器指定的名字 (一个绝对的字符串; 他只是一个代号)， value 是一个评估器对象:
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
estimators = [('linear_pca', PCA()), ('kernel_pca', KernelPCA())]
combined = FeatureUnion(estimators)
# 如 Pipeline, 单独的步骤可能用set_params替换 ,并设置为drop来跳过:
combined.set_params(kernel_pca='drop')
