# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/15 21:16
# @Function: 5.2.3. 文本特征提取
# 
# 1. 话语表示
# 原始数据，符号文字序列不能直接传递给算法，因为它们大多数要求具有固定长度的数字矩阵特征向量，而不是具有可变长度的原始文本文档。
# 解决方法：
#   令牌化（tokenizing） 对每个可能的词令牌分成字符串并赋予整数形的id，例如通过使用空格和标点符号作为令牌分隔符。
#   统计（counting） 每个词令牌在文档中的出现次数。
#   标准化（normalizing） 在大多数的文档 / 样本中，可以减少重要的次令牌的出现次数的权重。。
#
# 在该方案中，特征和样本定义如下：
#   每个单独的令牌发生频率（标准化或不标准化）被视为一个特征。
#   给定文档中所有的令牌频率向量被看做一个多元sample样本。
# 因此，文本的集合可被表示为矩阵形式，每行对应一条文本，每列对应每个文本中出现的词令牌(如单个词)。


# 2. 稀疏
# 由于大多数文本文档通常只使用文本词向量全集中的一个小子集，所以得到的矩阵将具有许多特征值为零（通常大于99％）。


# 3. 常见 Vectorizer 使用方法
# 类 CountVectorizer 在单个类中实现了 tokenization （词语切分）和 occurrence counting （出现频数统计）:
# 使用停止词


# 4. Tf–idf 项加权


# 5. 解码文本文件
# 文本由字符组成，但文件由字节组成。字节转化成字符依照一定的编码(encoding)方式。 为了在Python中的使用文本文档，这些字节必须被 解码


# 6. 应用和实例
#     监督学习：文档分类器
#     无监督学习：使用k-means聚类文本文档


# 7. 词语表示的限制






