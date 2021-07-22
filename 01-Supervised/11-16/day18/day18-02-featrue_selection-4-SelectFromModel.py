# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/22 10:21
# @Function:使用 SelectFromModel 选取特征 https://www.scikitlearn.com.cn/0.21.3/14/#1134-selectfrommodel
# SelectFromModel 是一个 meta-transformer（元转换器） ，
# 它可以用来处理任何带有 coef_ 或者 feature_importances_ 属性的训练之后的评估器。
# 如果相关的coef_ 或者 featureimportances 属性值低于预先设置的阈值，这些特征将会被认为不重要并且移除掉。
# 除了指定数值上的阈值之外，还可以通过给定字符串参数来使用内置的启发式方法找到一个合适的阈值。

"""
1. 基于L1的特征提取
    使用 L1 正则化的线性模型会得到稀疏解：他们的许多系数为 0。
    当目标是降低使用另一个分类器的数据集的维度， 它们可以与 feature_selection.SelectFromModel 一起使用来选择非零系数。
    特别的，可以用于此目的的稀疏评估器包括：
        用于回归的 linear_model.Lasso , 以及用于分类的 linear_model.LogisticRegression 和 svm.LinearSVC

    在 SVM 和逻辑回归中，参数 C 是用来控制稀疏性的：小的 C 会导致少的特征被选择。
    使用 Lasso，alpha 的值越大，越少的特征会被选择。

L1-recovery 和 compressive sensing（压缩感知）

当选择了正确的 alpha 值以后，假设特定的条件可以被满足的话, Lasso 可以仅通过少量观察点便恢复完整的非零特征
特别的，数据量需要 “足够大” ，不然 L1 模型的表现将缺乏保障。 
“足够大” 的定义取决于非零系数的个数、特征数量的对数值、噪音的数量、非零系数的最小绝对值、 以及设计矩阵（design maxtrix） X 的结构。
特征矩阵必须有特定的性质，如数据不能过度相关。

关于如何选择 alpha 值没有固定的规则。
alpha 值可以通过交叉验证来确定（ LassoCV 或者 LassoLarsCV ），
尽管这可能会导致欠惩罚的模型：包括少量的无关变量对于预测值来说并非致命的。
相反的， BIC（ LassoLarsIC ）倾向于给定高 alpha 值。
"""

"""
2. 基于 Tree（树）的特征选取
基于树的 estimators可以用来计算特征的重要性，然后可以消除不相关的特征


"""
