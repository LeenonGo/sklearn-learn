# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/5 13:32
# @Function: RANSAC  随机抽样一致性算法（RANdom SAmple Consensus）
# https://www.scikitlearn.com.cn/0.21.3/2/#11152-ransac-random-sample-consensus
# 利用全体数据中局内点（inliers）的一个随机子集拟合模型。

"""
算法细节

每轮迭代执行：
    1. 原始数据中抽样 min_samples 数量的随机样本，检查数据是否合法（见 is_data_valid ）
    2. 用一个随机子集拟合模型（ base_estimator.fit ）。检查模型是否合法
    3. 计算预测模型的残差.将全体数据分成局内点和离群点.绝对残差小于 residual_threshold 的全体数据认为是局内点。
    4. 若局内点样本数最大，保存当前模型为最佳模型。规定仅当数值大于当前最值时认为是最佳模型。
上述步骤或者迭代到最大次数（ max_trials ），或者某些终止条件满足时停下

"""

