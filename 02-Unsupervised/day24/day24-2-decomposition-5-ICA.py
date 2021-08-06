# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/6 22:07
# @Function: 独立成分分析（ICA）  https://www.scikitlearn.com.cn/0.21.3/24/#255-ica
# 

# 独立分量分析将多变量信号分解为独立性最强的加性子组件。 它通过 Fast ICA 算法在 scikit-learn 中实现。
# ICA 通常不用于降低维度，而是用于分离叠加信号。
# 由于 ICA 模型不包括噪声项，因此要使模型正确，必须使用白化(whitening)。
# 这可以在内部使用 whiten 参数或手动使用 PCA 的一种变体。

# 示例：day05-05.py

