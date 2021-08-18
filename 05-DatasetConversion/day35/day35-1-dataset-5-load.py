# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/18 18:24
# @Function:
#

# svmlight或libsvm格式的数据集
from sklearn.datasets import load_svmlight_file, load_svmlight_files

X_train, y_train = load_svmlight_file("/path/to/train_dataset.txt")
# 加载多个
X_train, y_train, X_test, y_test = load_svmlight_files(("/path/to/train_dataset.txt", "/path/to/test_dataset.txt"))
# svmlight / libsvm 格式的公共数据集 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

# 从openml.org下载数据集

# openml.org是一个用于机器学习数据和实验的公共存储库，它允许每个人上传开放的数据集。


# 从外部数据集加载
