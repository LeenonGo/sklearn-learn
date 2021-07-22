# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/22 15:42
# @Function: 概率校准 https://www.scikitlearn.com.cn/0.21.3/17/
#

"""
精确校准的分类器是概率分类器,其可以将 predict_proba 方法的输出直接解释为 confidence level（置信度级别）

良好校准的分类器是对于（预测概率方法的输出可以直接解释为置信水平的）概率分类器，。
例如，一个经过良好校准的（二元）分类器应该对样本进行分类，以便在其预测概率值接近0.8的样本中，大约80%的样本实际上属于正类。
LogisticRegression返回校准良好的预测，因为它直接优化了损失。相反，其他方法返回有偏差的概率，每个方法有不同的偏差：
    · 高斯贝叶斯倾向于将概率推到 0或 1（注意直方图中的计数）。
         这主要是因为它假设特征在给定类的情况下是有条件独立的，而在这个包含2个冗余特征的数据集中情况并非如此。
    · RandomForestClassifier显示了相反的行为：直方图显示的峰值概率约为0.2和0.9，而接近0或1的概率非常罕见。
        因为基础模型中的方差会使预测值偏离这些值，这些预测值应该接近0或1。
        所以像 bagging和随机森林法这样的方法从一组基本模型中平均预测，很难在0和1附近做出预测
        由于预测被限制在区间[0,1]，方差引起的误差往往是接近0和1的单边误差。
        例如，如果一个模型应该预测一个案例的p=0，那么bagging可以实现这一点的唯一方法就是所有bagging树都预测0。
        如果我们给bagging平均值超过的树添加噪声，这种噪声会导致一些树在这种情况下预测值大于0，从而使bagging集合的平均预测值远离0。
        我们在随机林中观察到这种效应最为明显，因为随机林训练的基水平树由于特征子集的存在而具有较高的方差。
        结果，校准曲线显示出特征的sigmoid形状，表明分类器可以更信任其“直觉”，并且返回概率通常接近0或1。
    · （SVC）显示了一条更为S形的曲线作为RandomForestClassifier，
        这对于关注接近决策边界（支持向量）的硬样本的最大margin方法是典型的。

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
np.random.seed(0)

X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                    n_informative=2, n_redundant=2)

train_samples = 100  # Samples used for training the models

X_train = X[:train_samples]
X_test = X[train_samples:]
y_train = y[:train_samples]
y_test = y[train_samples:]

# Create classifiers
lr = LogisticRegression()
gnb = GaussianNB()
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier()

# #############################################################################
# Plot calibration plots

plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
                  (svc, 'Support Vector Classification'),
                  (rfc, 'Random Forest')]:
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

    # 计算校准曲线的真实概率和预测概率
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (name,))
    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.show()
