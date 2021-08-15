# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/15 20:57
# @Function: 5.2.1. 从字典类型加载特征
# 类 DictVectorizer 可用于将标准的Python字典（dict）对象列表的要素数组转换为 scikit-learn 估计器使用的 NumPy/SciPy 表示形式。
# 虽然 Python 的处理速度不是特别快，但 Python 的 dict 优点是使用方便，稀疏（不需要存储的特征），并且除了值之外还存储特征名称。
# 类 DictVectorizer 实现了 “one-of-K” 或 “one-hot” 编码，用于分类（也称为标称，离散）特征。
# 分类功能是 “属性值” 对，其中该值被限制为不排序的可能性的离散列表（例如主题标识符，对象类型，标签，名称…）。


from sklearn.feature_extraction import DictVectorizer

# 城市” 是一个分类属性，而 “温度” 是传统的数字特征:
measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Francisco', 'temperature': 18.},
    {'city': 'London', 'temperature': 35.}
]
vec = DictVectorizer()
v= vec.fit_transform(measurements).toarray()
print(v)
n = vec.get_feature_names()
print(n)


















