# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/17 13:03
# @Function:
# 5.8.2. 线性核函数
# 函数 linear_kernel 是计算线性核函数, 也就是一种在 degree=1 和 coef0=0 (同质化) 情况下的 polynomial_kernel 的特殊形式.

# 5.8.3. 多项式核函数
# 函数polynomial_kernel计算两个向量的d次方的多项式核函数. 多项式核函数代表着两个向量之间的相似度.
# 概念上来说，多项式核函数不仅考虑相同维度还考虑跨维度的向量的相似度。

# 5.8.4. Sigmoid 核函数
# 函数 sigmoid_kernel 计算两个向量之间的S型核函数. S型核函数也被称为双曲切线或者 多层感知机(因为在神经网络领域，它经常被当做激活函数).

# 5.8.5. RBF 核函数
# 函数 rbf_kernel 计算计算两个向量之间的径向基函数核 (RBF) 。

# 5.8.6. 拉普拉斯核函数
# 函数 laplacian_kernel 是一种径向基函数核的变体

# 5.8.7. 卡方核函数
# 在计算机视觉应用中训练非线性支持向量机时，卡方核函数是一种非常流行的选择.
# 它能以 chi2_kernel 计算然后将参数kernel=”precomputed”传递到 sklearn.svm.SVC :
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
X = [[0, 1], [1, 0], [.2, .8], [.7, .3]]
y = [0, 1, 0, 1]
K = chi2_kernel(X, gamma=.5)
print(K)
svm = SVC(kernel='precomputed').fit(K, y)
print(svm.predict(K))
