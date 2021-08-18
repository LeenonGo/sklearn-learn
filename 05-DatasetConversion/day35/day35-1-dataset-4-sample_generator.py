# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/18 18:11
# @Function: 6.4. 样本生成器
# scikit-learn 包括各种随机样本的生成器，可以用来建立可控制的大小和复杂性人工数据集。

# 6.4.1. 分类和聚类生成器
# 1. 单标签
# 这些生成器将产生一个相应特征的离散矩阵。
# make_blobs 和 make_classification 通过分配每个类的一个或多个正态分布的点的群集创建的多类数据集。
# make_blobs 对于中心和各簇的标准偏差提供了更好的控制，可用于演示聚类。
# make_classification 专门通过引入相关的，冗余的和未知的噪音特征；将高斯集群的每类复杂化；在特征空间上进行线性变换。

# 2. 多标签
# make_multilabel_classification 生成多个标签的随机样本，反映从a mixture of topics（一个混合的主题）中引用a bag of words （一个词袋）
# 每个文档的主题数是基于泊松分布随机提取的，同时主题本身也是从固定的随机分布中提取的。

# 3. 二分聚类
# make_biclusters和 make_checkerboard


# 6.4.2. 回归生成器
# make_regression 产生的回归目标作为一个可选择的稀疏线性组合的具有噪声的随机的特征。它的信息特征可能是不相关的或低秩（少数特征占大多数的方差）。


# 6.4.3. 流形学习生成器


