# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/7 22:33
# @Function: 2.6.2. 收缩协方差  https://www.scikitlearn.com.cn/0.21.3/25/#262
#

# 1. 基本收缩
# 尽管是协方差矩阵的无偏估计,最大似然估计不是协方差矩阵的特征值的一个很好的估计,所以从反演(矩阵的求逆过程)得到的精度矩阵是不准确的
# 有时，甚至出现因矩阵元素地特性,经验协方差矩阵不能求逆。为了避免这样的反演问题，引入了经验协方差矩阵的一种变换方式：shrinkage 。
# 在 scikit-learn 中，该变换（使用用户定义的收缩系数） 可以直接应用于使用 shrunk_covariance 方法预先计算的协方差。
# 此外，协方差的收缩估计可以通过 ShrunkCovariance 对象的 ShrunkCovariance.fit 方法拟合到数据中。

# 在数学上，这种收缩在于减少经验协方差矩阵的最小和最大特征值之间的比率 可以通过简单地根据给定的偏移量移动每个特征值来完成，
# 这相当于找到协方差矩阵的l2惩罚的最大似然估计器


# 2. Ledoit-Wolf 收缩
# 计算最优的收缩系数 \alpha ，它使得估计协方差和实际协方差矩阵之间的均方误差(Mean Squared Error)进行最小化。


# 3. Oracle 近似收缩   OAS
# 数据为高斯分布的假设下，旨在 产生比 Ledoit-Wolf 公式具有更小均方误差的收缩系数。

"""
示例：收缩协方差估计
当使用协方差估计时，通常的方法是使用最大似然估计。如EmpiricalCovariance
它是无偏的，即当给定许多观测值时，它收敛于真实（总体）协方差。
然而，为了减少其方差，对其进行正则化也是有益的。这反过来又引入了一些偏差。

这个例子说明了收缩协方差估计中使用的简单正则化。
特别是，它着重于如何设置正则化量，即如何选择偏差-方差权衡。

这里我们比较3种方法：
    通过根据潜在收缩参数网格交叉验证三个折叠的可能性来设置参数。
    Ledoit和Wolf提出了一个用于计算渐近最优正则化参数（最小化MSE准则）的闭合公式，得到了LedoitWolf协方差估计。
    Chen等人提出的Ledoit-Wolf收缩的一种改进，即OAS。在假设数据是高斯的情况下，尤其是对于小样本，其收敛性明显更好。

为了量化估计误差，我们绘制了不同收缩参数值的不可见数据的可能性。我们还通过交叉验证或LedoitWolf和OAS估计来显示选择。

请注意，最大似然估计对应于无收缩，因此性能较差。Ledoit-Wolf估计的性能非常好，因为它接近最优值，并且计算成本不高。
在本例中，OAS估计值稍微远一点。有趣的是，这两种方法都优于交叉验证，交叉验证的计算成本非常高。
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg

from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance, log_likelihood, empirical_covariance
from sklearn.model_selection import GridSearchCV


# #############################################################################
np.random.seed(42)
# Generate sample data
n_features, n_samples = 40, 20
base_X_train = np.random.normal(size=(n_samples, n_features))
base_X_test = np.random.normal(size=(n_samples, n_features))

# Color samples
coloring_matrix = np.random.normal(size=(n_features, n_features))
X_train = np.dot(base_X_train, coloring_matrix)
X_test = np.dot(base_X_test, coloring_matrix)

# #############################################################################
# Compute the likelihood on test data

# 涵盖一系列可能的收缩系数值
shrinkages = np.logspace(-2, 0, 30)
negative_logliks = [-ShrunkCovariance(shrinkage=s).fit(X_train).score(X_test) for s in shrinkages]

# under the ground-truth model, which we would not have access to in real settings
real_cov = np.dot(coloring_matrix.T, coloring_matrix)
emp_cov = empirical_covariance(X_train)
loglik_real = -log_likelihood(emp_cov, linalg.inv(real_cov))

# #############################################################################
# Compare different approaches to setting the parameter

# GridSearch for an optimal shrinkage coefficient
tuned_parameters = [{'shrinkage': shrinkages}]
cv = GridSearchCV(ShrunkCovariance(), tuned_parameters)
cv.fit(X_train)

# Ledoit-Wolf optimal shrinkage coefficient estimate
lw = LedoitWolf()
loglik_lw = lw.fit(X_train).score(X_test)

# OAS coefficient estimate
oa = OAS()
loglik_oa = oa.fit(X_train).score(X_test)

# #############################################################################
# Plot results
fig = plt.figure()
plt.title("Regularized covariance: likelihood and shrinkage coefficient")
plt.xlabel('Regularization parameter: shrinkage coefficient')
plt.ylabel('Error: negative log-likelihood on test data')
# range shrinkage curve
plt.loglog(shrinkages, negative_logliks, label="Negative log-likelihood")

plt.plot(plt.xlim(), 2 * [loglik_real], '--r', label="Real covariance likelihood")

# adjust view
lik_max = np.amax(negative_logliks)
lik_min = np.amin(negative_logliks)
ymin = lik_min - 6. * np.log((plt.ylim()[1] - plt.ylim()[0]))
ymax = lik_max + 10. * np.log(lik_max - lik_min)
xmin = shrinkages[0]
xmax = shrinkages[-1]
# LW likelihood
plt.vlines(lw.shrinkage_, ymin, -loglik_lw, color='magenta', linewidth=3, label='Ledoit-Wolf estimate')
# OAS likelihood
plt.vlines(oa.shrinkage_, ymin, -loglik_oa, color='purple', linewidth=3, label='OAS estimate')
# best CV estimator likelihood
plt.vlines(cv.best_estimator_.shrinkage, ymin, -cv.best_estimator_.score(X_test), color='cyan',
           linewidth=3, label='Cross-validation best estimate')

plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)
plt.legend()

plt.show()


