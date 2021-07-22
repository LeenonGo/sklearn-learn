# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/22 10:45
# @Function: 半监督学习 https://www.scikitlearn.com.cn/0.21.3/15/
# 适用于在训练数据上的一些样本数据没有贴上标签的情况
# sklearn.semi_supervised 中的半监督估计,
# 能够利用这些附加的未标记数据来更好地捕获底层数据分布的形状，并将其更好地类推到新的样本。
# 当我们有非常少量的已标签化的点和大量的未标签化的点时，这些算法表现均良好。
"""
标签传播：
表示半监督图推理算法的几个变体。特性：
    * 可用于分类和回归任务
    * 使用内核方法将数据投影到备用维度空间
模型：LabelPropagation 和 LabelSpreading
两者都通过在输入的 dataset（数据集）中的所有 items（项）上构建 similarity graph （相似图）来进行工作。

"""
# 示例
# LabelPropagation学习的例子展示了一个复杂的内部结构“流形学习”。
# 外圈标记为“红色”，内圈标记为“蓝色”。
# 因为两个标签组都位于各自不同的形状内，所以我们可以看到标签正确地围绕圆传播。
import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelSpreading
from sklearn.datasets import make_circles

# generate ring with inner box
n_samples = 200
X, y = make_circles(n_samples=n_samples, shuffle=False)  # 在2d中生成一个包含一个小圆的大圆。
outer, inner = 0, 1
labels = np.full(n_samples, -1.)
labels[0] = outer
labels[-1] = inner

# #############################################################################
# Learn with LabelSpreading
label_spread = LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(X, labels)

# #############################################################################
# Plot output labels
output_labels = label_spread.transduction_
plt.figure(figsize=(8.5, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[labels == outer, 0], X[labels == outer, 1], color='navy',  marker='s', lw=0, label="outer labeled", s=10)
plt.scatter(X[labels == inner, 0], X[labels == inner, 1], color='c', marker='s', lw=0, label='inner labeled', s=10)
plt.scatter(X[labels == -1, 0], X[labels == -1, 1], color='darkorange', marker='.', label='unlabeled')
plt.legend(scatterpoints=1, shadow=False, loc='upper right')
plt.title("Raw data (2 classes=outer and inner)")

plt.subplot(1, 2, 2)
output_label_array = np.asarray(output_labels)
outer_numbers = np.where(output_label_array == outer)[0]
inner_numbers = np.where(output_label_array == inner)[0]
plt.scatter(X[outer_numbers, 0], X[outer_numbers, 1], color='navy', marker='s', lw=0, s=10, label="outer learned")
plt.scatter(X[inner_numbers, 0], X[inner_numbers, 1], color='c',  marker='s', lw=0, s=10, label="inner learned")
plt.legend(scatterpoints=1, shadow=False, loc='upper right')
plt.title("Labels learned with Label Spreading (KNN)")

plt.subplots_adjust(left=0.07, bottom=0.07, right=0.93, top=0.92)
plt.show()

