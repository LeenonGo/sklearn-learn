# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/14 8:42
# @Function:
# 高斯朴素贝叶斯 https://www.scikitlearn.com.cn/0.21.3/10/#191
#   实现了运用于分类的高斯朴素贝叶斯算法
#   特征的可能性(即概率)假设为高斯分布
#   参数 \sigma_y 和 \mu_y 使用极大似然法估计
#
#
# 多项分布朴素贝叶斯：https://www.scikitlearn.com.cn/0.21.3/10/#192
#   实现了服从多项分布数据的朴素贝叶斯算法
#
#
# 补充朴素贝叶斯 https://www.scikitlearn.com.cn/0.21.3/10/#193  ComplementNB
#   CNB是标准多项式朴素贝叶斯(MNB)算法的一种改进，特别适用于不平衡数据集。

#
# 伯努利朴素贝叶斯： BernoulliNB  https://www.scikitlearn.com.cn/0.21.3/10/#194
# 实现了用于多重伯努利分布数据的朴素贝叶斯训练和分类算法，即有多个特征，
# 但每个特征 都假设是一个二元 (Bernoulli, boolean) 变量。


# 基于外存的朴素贝叶斯模型拟合： https://www.scikitlearn.com.cn/0.21.3/10/#195
#   朴素贝叶斯模型可以解决整个训练集不能导入内存的大规模分类问题。
#   MultinomialNB, BernoulliNB, 和 GaussianNB 实现了 partial_fit 方法，可以动态的增加数据
#   与 fit 方法不同，首次调用 partial_fit 方法需要传递一个所有期望的类标签的列表。
