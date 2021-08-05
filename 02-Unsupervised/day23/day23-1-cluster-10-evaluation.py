# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/5 23:20
# @Function: https://www.scikitlearn.com.cn/0.21.3/22/#2310
# 
from sklearn import metrics
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]

# ########### 1. 调整后的兰德指数  用于测量两个簇标签分配的值的 相似度 #################
s = metrics.adjusted_rand_score(labels_true, labels_pred)
print(s)
# 在预测的标签列表中重新排列 0 和 1, 把 2 重命名为 3, 得到相同的得分
labels_pred = [1, 1, 0, 0, 3, 3]
s2 = metrics.adjusted_rand_score(labels_true, labels_pred)
print(s2)
# 对称性
s3 = metrics.adjusted_rand_score(labels_pred, labels_true)
# 不良标签  得分是负数 或 接近于 0.0 分
labels_true = [0, 1, 2, 0, 3, 4, 5, 1]
labels_pred = [1, 1, 0, 0, 2, 2, 2, 2]
s4 = metrics.adjusted_rand_score(labels_true, labels_pred)
# ########### ################## ########### #################


# ########### 2. 基于互信息(mutual information)的得分  测量两个标签分配的 一致性 #################
print(metrics.adjusted_mutual_info_score(labels_true, labels_pred))
labels_pred = [1, 1, 0, 0, 3, 3]
# 在预测的标签列表中重新排列 0 和 1, 把 2 重命名为 3, 得到相同的得分:
print(metrics.adjusted_mutual_info_score(labels_true, labels_pred))
# mutual_info_score, adjusted_mutual_info_score 和 normalized_mutual_info_score 是对称的
# ########### ################## ########### #################


# ########### 3. 同质性，完整性和 V-measure #################
# 同质性(homogeneity): 每个簇只包含一个类的成员
# 完整性(completeness): 给定类的所有成员都分配给同一个簇。
# 这两个分数越高越好
h = metrics.homogeneity_score(labels_true, labels_pred)
c = metrics.completeness_score(labels_true, labels_pred)
# 称为 V-measure 的调和平均数(harmonic mean)由v_measure_score函数计算
# beta默认值为1.0，但如果beta值小于1，则为:
v1 = metrics.v_measure_score(labels_true, labels_pred, beta=0.6)
# 更大的beta权重将提高同质性，当使用大于1的beta值时   (更大的beta权重将提高完整性。)
v2 = metrics.v_measure_score(labels_true, labels_pred, beta=1.8)
# 同质性, 完整性 and V-measure 可以使用 homogeneity_completeness_v_measure进行计算
a = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
# ########### ################## ########### #################


# ########### 4. Fowlkes-Mallows得分    成对的准确率和召回率的几何平均值: #################
f = metrics.fowlkes_mallows_score(labels_true, labels_pred)   # 得分范围为 0 到 1。较高的值表示两个簇之间的良好相似性
# ########### ################## ########### #################


# ########### 5. Silhouette 系数 #################
# 如果不知道真实簇标签，则必须使用模型本身进行度量。
# Silhouette 系数是一个这样的评估例子，其中较高的 Silhouette 系数得分和能够更好定义的聚类的模型相关联。
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target
# 一组样本的 Silhouette 系数可以作为其中每个样本的 Silhouette 系数的平均值。
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_
s = metrics.silhouette_score(X, labels, metric='euclidean')  # 在正常使用情况下，将 Silhouette 系数应用于聚类结果的分析。
# ########### ################## ########### #################


# ########### 6. Calinski-Harabaz 指数 #################
# 如果不知道真实簇标签，则可以使用 Calinski-Harabaz 指数或被称为方差比准则(Variance Ratio Criterion)-来评估模型，
# 其中较高的 Calinski-Harabaz 的得分和能够更好定义的聚类的模型相关联。
c = metrics.calinski_harabaz_score(X, labels)
# 当簇密集且分离较好时，分数更高，这关联到了簇的标准概念。
# 得分计算很快。
# ########### ################## ########### #################


# ########### 7. Davies-Bouldin Index #################
# 如果不知道真实簇标签，则可以使用 Davies-Bouldin Index（简称: DBI）去度量模型
d = metrics.davies_bouldin_score(X, labels)
# ########### ################## ########### #################


# ########### 8. Contingency Matrix #################
# 列联矩阵（Contingency Matrix）记录了每个真实/预测簇对之间的交叉基数。
# 列联矩阵为所有的聚合度量提供了足量的统计数据，而其中样本都是独立和相同分布，并不考虑没有被聚合的实例。
from sklearn.metrics.cluster import contingency_matrix
x = ["a", "a", "a", "b", "b", "b"]
y = [0, 0, 1, 1, 2, 2]
c = contingency_matrix(x, y)
# 输出array的第一行 意味着有三个样本的真实簇是'a'.
# 而在这三个样本中, 两个的预测簇是 0,一个是 1, 没有一个是 2。
# 而第二行意味着有三个样本的真实簇 是'b'。 而其中，没有一个样本的预测簇是 0, 一个是 1, 两个是 2。












