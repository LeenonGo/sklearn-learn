# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/28 16:03
# @Function: 不同的标记分配策略  https://www.scikitlearn.com.cn/0.21.3/22/#2351
# 
# SpectralClustering中  assign_labels 参数代表着可以使用不同的分配策略。
#   kmeans 可以匹配更精细的数据细节，但是可能更加不稳定。
#   需要控制 random_state 以复现运行的结果 ，因为它取决于随机初始化。
#   使用 discretize 策略是 100% 可以复现的，但是它往往会产生相当均匀的几何形状的闭合块。
#
#
# 谱聚类还可以通过谱嵌入对图进行聚类。在这种情况下，关联矩阵(affinity matrix) 是图形的邻接矩阵，
# 谱聚类可以由 affinity=’precomputed’ 进行初始化。


# 参考资料：https://www.cnblogs.com/pinard/p/6221564.html
