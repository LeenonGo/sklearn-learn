# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/22 16:18
# @Function:
# 对具有20个特征的100.000个样本（其中一个用于模型拟合）进行二元分类的人造数据集进行以下实验.
# 在 20个 特征中，只有 2 个是信息量, 10 个是冗余的.
# 该图显示了使用逻辑回归获得的估计概率, 线性支持向量分类器（SVC）和具有 sigmoid 校准和 sigmoid 校准的线性 SVC.
# 校准性能使用 Brier score的 brier_score_loss 来计算（分数越小越好）, 请看下面的图例
"""
本例演示如何显示预测概率的校准程度以及如何校准未校准的分类器。
第一幅图显示了使用logistic回归、高斯朴素贝叶斯和高斯朴素贝叶斯（同时使用等式校准和sigmoid校准）获得的估计概率。
用图例中报告的Brier分数（越小越好）评估校准性能。
我们可以在这里观察到，logistic回归是很好的校准，而原始高斯朴素贝叶斯执行非常糟糕。
这是因为冗余特征违反了特征独立性的假设，导致分类器过于自信，这一点由典型的转置sigmoid曲线表示。

用等式回归校正高斯朴素贝叶斯概率可以解决这个问题，这可以从几乎对角的校正曲线上看出。
Sigmoid校准也略微改善了brier评分，尽管不如非参数等式回归那么强。
这可以归因于我们有大量的校准数据，这样就可以利用非参数模型更大的灵活性。

第二幅图显示了线性支持向量分类器（LinearSVC）的校准曲线。
LinearSVC表现出与高斯朴素贝叶斯相反的行为：
    校准曲线具有sigmoid曲线，这是欠自信分类器的典型特征。
    在LinearSVC的情况下，这是由hinge损失的边缘特性引起的，这使得模型可以聚焦于接近决策边界的硬样本（支持向量）。

两种校准都可以解决这个问题，并产生几乎相同的结果。
这表明sigmoid校准可以处理基本分类器的校准曲线是sigmoid（例如，对于LinearSVC）而不是转置sigmoid（例如，高斯朴素贝叶斯）的情况。
 然而, 非参数等渗校准模型没有这样强大的假设, 并且可以处理任何形状, 只要有足够的校准数据.
 通常，在校准曲线为 sigmoid 且校准数据有限的情况下, sigmoid 校准是优选的,
 而对于非 sigmoid 校准曲线和大量数据可用于校准的情况，等渗校准是优选的.
"""

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split

X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                    n_informative=2, n_redundant=10,
                                    random_state=42)  # 2 个是信息量, 10 个是冗余的

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=42)


def plot_calibration_curve(est, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1.)

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'), (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % clf_score)
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % (name, clf_score))
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()


# Plot calibration curve for Gaussian Naive Bayes
plot_calibration_curve(GaussianNB(), "Naive Bayes", 1)

# Plot calibration curve for Linear SVC
plot_calibration_curve(LinearSVC(max_iter=10000), "SVC", 2)

plt.show()
