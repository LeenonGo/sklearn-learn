# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/1 16:08
# @Function: logistic 回归：https://www.scikitlearn.com.cn/0.21.3/2/#1111-logistic
# 解决分类问题
# scikit-learn 中 logistic 回归在 LogisticRegression 类中实现了
# 二分类（binary）、一对多分类（one-vs-rest）及多项式 logistic 回归，并带有可选的 L1 和 L2 正则化。
# 默认情况下使用L2正则化
# 正则化：提升数值稳定性
# 在 LogisticRegression 类中实现了这些优化算法:
#   liblinear：应用坐标下降算法
#   newton-cg， lbfgs， sag：只支持 L2罚项以及无罚项，对某些高维数据收敛更快只支持 L2罚项以及无罚项，对某些高维数据收敛更快
#   sag: 基于平均随机梯度下降算法.在大数据集上的表现更快
#   saga: sag 的一类变体,支持非平滑（non-smooth）的 L1 正则选项.多用于解决稀疏多项式 logistic 回归。
#       唯一支持弹性网络正则选项的求解器
#   lbfgs: 推荐用于较小的数据集。属于准牛顿法


