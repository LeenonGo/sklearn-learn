# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/22 16:44
# @Function: 三级分类的概率校正
#
# CalibratedClassifierCV 也可以处理涉及两个以上类的分类任务, 如果基本估计器可以这样做的话.
# 在这种情况下, 分类器是以一对一的方式分别对每个类进行校准.
# 当预测未知数据的概率时, 分别预测每个类的校准概率. 由于这些概率并不总是一致, 因此执行后处理以使它们归一化.
"""
下一个图像说明了 Sigmoid 校准如何改变 3 类分类问题的预测概率.
说明是标准的 2-simplex，其中三个角对应于三个类.
箭头从未校准分类器预测的概率向量指向在保持验证集上的 sigmoid 校准之后由同一分类器预测的概率向量.
颜色表示实例的真实类（red: class 1, green: class 2, blue: class 3）.

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
np.random.seed(0)

X, y = make_blobs(n_samples=2000, n_features=2, centers=3, random_state=42, cluster_std=5.0)
X_train, y_train = X[:600], y[:600]
X_valid, y_valid = X[600:1000], y[600:1000]
X_train_valid, y_train_valid = X[:1000], y[:1000]
X_test, y_test = X[1000:], y[1000:]

# 首先，我们将训练一个具有25个基估计（树）的RandomForestClassifier在连接的训练和验证数据（1000个样本）上。这是未校准的分类器。
clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train_valid, y_train_valid)

# 为了训练被校准的分类器，从相同的randomforest分类器开始，但是只使用训练数据子集（600个样本）训练它，然后校准，
# 使用method='sigmoid' 在两阶段过程中使用有效数据子集（400个样本）。
clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train, y_train)
cal_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
cal_clf.fit(X_valid, y_valid)

# 比较概率
plt.figure(figsize=(10, 10))
colors = ["r", "g", "b"]

clf_probs = clf.predict_proba(X_test)
cal_clf_probs = cal_clf.predict_proba(X_test)
for i in range(clf_probs.shape[0]):
    xs = clf_probs[i, 0]
    ys = clf_probs[i, 1]
    xd = cal_clf_probs[i, 0] - xs  # 校验过程  见下方注解
    yd = cal_clf_probs[i, 1] - ys
    plt.arrow(xs, ys, xd, yd, color=colors[y_test[i]], head_width=1e-2)

# 在每个顶点绘制完美的预测
plt.plot([1.0], [0.0], 'ro', ms=20, label="Class 1")
plt.plot([0.0], [1.0], 'go', ms=20, label="Class 2")
plt.plot([0.0], [0.0], 'bo', ms=20, label="Class 3")

# Plot boundaries of unit simplex  三角边
plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 'k', label="Simplex")

# Annotate points 6 points around the simplex, and mid point inside simplex
plt.annotate(r'($\frac{1}{3}$, $\frac{1}{3}$, $\frac{1}{3}$)',
             xy=(1.0/3, 1.0/3), xytext=(1.0/3, .23), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.plot([1.0/3], [1.0/3], 'ko', ms=5)
plt.annotate(r'($\frac{1}{2}$, $0$, $\frac{1}{2}$)',
             xy=(.5, .0), xytext=(.5, .1), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($0$, $\frac{1}{2}$, $\frac{1}{2}$)',
             xy=(.0, .5), xytext=(.1, .5), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($\frac{1}{2}$, $\frac{1}{2}$, $0$)',
             xy=(.5, .5), xytext=(.6, .6), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($0$, $0$, $1$)',
             xy=(0, 0), xytext=(.1, .1), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($1$, $0$, $0$)',
             xy=(1, 0), xytext=(1, .1), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')
plt.annotate(r'($0$, $1$, $0$)',
             xy=(0, 1), xytext=(.1, 1), xycoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='center')

plt.grid(False)
for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    plt.plot([0, x], [x, 0], 'k', alpha=0.2)
    plt.plot([0, 0 + (1-x)/2], [x, x + (1-x)/2], 'k', alpha=0.2)
    plt.plot([x, x + (1-x)/2], [0, 0 + (1-x)/2], 'k', alpha=0.2)


plt.title("Change of predicted probabilities on test samples "
          "after sigmoid calibration")
plt.xlabel("Probability class 1")
plt.ylabel("Probability class 2")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
_ = plt.legend(loc="best")

# 在上图中，图形的每个顶点表示一个完全预测的类（例如，1，0，0）。
# 图形内的中点表示以相等的概率（即1/3、1/3、1/3）预测三个类别。
# 每个箭头从未校准的概率开始，以校准概率的箭头结束。箭头的颜色表示该测试样本的真实类。
# （
#   以校验值减去未校验值，可以用向量解释，向量 a 减去向量 b 表示从 b 的终点指向 a 的终点
#   xd = cal_clf_probs[i, 0] - xs  表示从未校验的值指向校验的值
#  ）
# 未校准的分类器对其预测过于自信，并且会导致大量的 log loss。
# 由于两个因素，校准后的校准分类器产生较低的对数损失。
# 首先，请注意上图中的箭头通常指向三角形的边，其中一个类的概率为0。
# 第二，大部分箭头指向真类，例如绿色箭头（真类为“绿色”的示例）通常指向绿色顶点。
# 这会减少过度自信的0预测概率，同时增加正确类的预测概率。因此，校准的分类器产生更准确的预测概率，从而产生更低的对数损失
# 我们可以通过比较未校准和校准分类器在1000个测试样本的预测上的对数损失来客观地说明这一点。
# 请注意，另一种方法是增加RandomForestClassifier的基估计数（树），这将导致类似的对数损失减少。

score = log_loss(y_test, clf_probs)
cal_score = log_loss(y_test, cal_clf_probs)

print("Log-loss of")
print(f" * uncalibrated classifier: {score:.3f}")
print(f" * calibrated classifier: {cal_score:.3f}")

# 最后，我们在2维图形形上生成一个可能的未校准概率网格，计算相应的校准概率并绘制每个概率的箭头。
# 箭头按最高未校准概率上色。这说明了学习的校准图：

plt.figure(figsize=(10, 10))
# Generate grid of probability values
p1d = np.linspace(0, 1, 20)
p0, p1 = np.meshgrid(p1d, p1d)
p2 = 1 - p0 - p1
p = np.c_[p0.ravel(), p1.ravel(), p2.ravel()]
p = p[p[:, 2] >= 0]

# 使用三个类的校准器来计算校准的概率
calibrated_classifier = cal_clf.calibrated_classifiers_[0]
prediction = np.vstack([calibrator.predict(this_p)
                        for calibrator, this_p in
                        zip(calibrated_classifier.calibrators, p.T)]).T

# Re-normalize the calibrated predictions to make sure they stay inside the
# simplex. This same renormalization step is performed internally by the
# predict method of CalibratedClassifierCV on multiclass problems.
# 重新标准化校准的预测，以确保它们留在图形内。同样的重整化步骤是由CCV对多类问题的预测方法在内部完成的。
prediction /= prediction.sum(axis=1)[:, None]

# 绘制由校准器引起的预测概率变化
for i in range(prediction.shape[0]):
    plt.arrow(p[i, 0], p[i, 1],
              prediction[i, 0] - p[i, 0], prediction[i, 1] - p[i, 1],
              head_width=1e-2, color=colors[np.argmax(p[i])])

# Plot the boundaries of the unit simplex
plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 'k', label="Simplex")

plt.grid(False)
for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    plt.plot([0, x], [x, 0], 'k', alpha=0.2)
    plt.plot([0, 0 + (1-x)/2], [x, x + (1-x)/2], 'k', alpha=0.2)
    plt.plot([x, x + (1-x)/2], [0, 0 + (1-x)/2], 'k', alpha=0.2)

plt.title("Learned sigmoid calibration map")
plt.xlabel("Probability class 1")
plt.ylabel("Probability class 2")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.show()


