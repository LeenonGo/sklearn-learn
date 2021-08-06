# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/6 22:00
# @Function: 小批量字典学习  https://www.scikitlearn.com.cn/0.21.3/24/#2533
# 
# MiniBatchDictionaryLearning 实现了更快、更适合大型数据集的字典学习算法，其运行速度更快，但准确度有所降低。
#
# 默认情况下，MiniBatchDictionaryLearning 将数据分成小批量，并通过在指定次数的迭代中循环使用小批量，以在线方式进行优化。
# 但是，目前它没有实现停止条件。
#
# 估计器还实现了 partial_fit, 它通过在一个小批处理中仅迭代一次来更新字典。
# 当在线学习的数据从一开始就不容易获得，或者数据超出内存时，可以使用这种迭代方法。



