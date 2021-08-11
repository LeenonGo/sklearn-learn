# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/11 17:21
# @Function: 3.3. 模型评估: 量化预测的质量  https://www.scikitlearn.com.cn/0.21.3/32/
# 

# 有 3 种不同的 API 用于评估模型预测的质量
#   Estimator score method（估计器得分的方法）:
#       Estimators（估计器）有一个 score（得分） 方法，为其解决的问题提供了默认的 evaluation criterion （评估标准）。
#   Scoring parameter（评分参数）:
#       Model-evaluation tools （模型评估工具）使用 cross-validation依靠 internal scoring strategy
#   Metric functions（指标函数）:
#       metrics 模块实现了针对特定目的评估预测误差的函数。

# --------------------------- 3.3.1. scoring 参数: 定义模型评估规则 ---------------------------------

# Model selection （模型选择）和 evaluation （评估）使用工具，
# 例如 model_selection.GridSearchCV 和 model_selection.cross_val_score ，
# 采用 scoring 参数来控制它们对 estimators evaluated （评估的估计量）应用的指标。

# 使用 scoring 参数指定一个 scorer object （记分对象），如'accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score',等

# 模块 sklearn.metrics 还公开了一组 measuring a prediction error （测量预测误差）的简单函数，给出了基础真实的数据和预测:
#   函数以 _score 结尾返回一个值来最大化，越高越好。
#   函数 _error 或 _loss 结尾返回一个值来 minimize （最小化），越低越好。
#       当使用 make_scorer 转换成 scorer object （记分对象）时，将 greater_is_better 参数设置为 False（默认为 True）


# ------------------------------------- 3.3.2. 分类指标 --------------------------------------
# sklearn.metrics 模块实现了几个 loss, score, 和 utility 函数来衡量 classification 性能。
# 某些 metrics可能需要 positive class，confidence values或 binary decisions values 的概率估计。
# 大多数的实现允许每个样本通过 sample_weight 参数为 overall score提供 weighted contribution （加权贡献）。

# 一些 metrics 基本上是为二分类任务定义的 (例如 f1_score, roc_auc_score) 。
# 在这些情况下，默认情况下仅评估positive label，假设默认情况下，positive label标记为 1
# 将 binary metric （二分指标）扩展为 multiclass （多类）或 multilabel （多标签）问题时，数据将被视为二分问题的集合，每个类都有一个。

# accuracy_score 函数计算 accuracy, 正确预测的分数（默认）或计数 (normalize=False)。
# 在多标签分类中，函数返回 subset accuracy。如果样本的整套预测标签与真正的标签组合匹配，则子集精度为 1.0; 否则为 0.0 。
# day29-1-ModelEvaluation-eg1.py

# balanced_accuracy_score函数计算 balanced accuracy, 它可以避免在不平衡数据集上作出夸大的性能估计。
# 在二分类情况下, balanced accuracy等价于sensitivity(true positive rate)和 specificity(真负率:true negative rate)的算术平均值,
# 或者ROC曲线下具有二元预测值的面积，而不是分数。
# 分类器在两个类上都表现的一样好，该函数就会退化为传统的准确率(即正确预测数量除以总的预测数量).
# 作为对比, 如果传统的准确率比较好，仅仅是因为分类器利用了一个不均衡测试集，此时balanced_accuracy将会近似地掉到1/n_classes

# cohen_kappa_score计算 Cohen’s kappa statistic（统计）。这个 measure旨在比较不同人工标注者的标签，而不是classifier与ground truth。

# confusion_matrix 函数通过计算 confusion matrix（混淆矩阵） 来 evaluates classification accuracy （评估分类的准确性）。

# 分类报告 classification_report 函数构建一个显示 main classification metrics （主分类指标）的文本报告。

# 汉明损失 hamming_loss 计算两组样本之间的 average Hamming loss （平均汉明损失）或者 Hamming distance（汉明距离） 。

# 精准，召回和 F-measures

# Jaccard 相似系数 score

# ....

# 更多的度量标准在示例中都有介绍过，或在后面的例子中具体用到再介绍


