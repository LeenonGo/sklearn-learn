# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/15 21:07
# @Function: 5.2.2. 特征哈希（相当于一种降维技巧）
# 类 FeatureHasher 是一种高速，低内存消耗的向量化方法，它使用了特征散列技术 ，或可称为 “散列法” （hashing trick）的技术。
# 以牺牲可检测性为代价，提高速度和减少内存的使用; 哈希表不记得输入特性是什么样的，没有 inverse_transform 办法。

# 由于散列函数可能导致（不相关）特征之间的冲突，因此使用带符号散列函数，并且散列值的符号确定存储在特征的输出矩阵中的值的符号。
# 这样，碰撞可能会抵消而不是累积错误，并且任何输出要素的值的预期平均值为零。
#
from sklearn.feature_extraction import FeatureHasher


def token_features(token, part_of_speech):
    if token.isdigit():
        yield "numeric"
    else:
        yield "token={}".format(token.lower())
        yield "token,pos={},{}".format(token, part_of_speech)
    if token[0].isupper():
        yield "uppercase_initial"
    if token.isupper():
        yield "all_uppercase"
    yield "pos={}".format(part_of_speech)


raw_X = (token_features(tok, pos_tagger(tok)) for tok in corpus)

hasher = FeatureHasher(input_type='string')
X = hasher.transform(raw_X)




