# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/13 9:06
# @Function: GPR 高斯过程回归  https://www.scikitlearn.com.cn/0.21.3/8/#171-gpr
# 
"""
使用GPR时，要先指定GP的先验：
    normalize_y=False，先验的均值通常假定为常数或者零
    normalize_y=True ，先验均值通常为训练数据的均值
    先验的方差通过传递 内核(kernel) 对象来指定

"""
