# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/7 22:57
# @Function: 3. 稀疏逆协方差   https://www.scikitlearn.com.cn/0.21.3/25/#263

# 协方差矩阵的逆矩阵，通常称为精度矩阵，它与部分相关矩阵成正比。它给出部分独立性关系。
# 换句话说，如果两个特征与其他特征有条件地独立，则精度矩阵中的对应系数将为零。
# 这就是为什么估计一个稀疏精度矩阵是有道理的：通过从数据中学习独立关系，可以得到更好的协方差矩阵估计。这被称为协方差选择
# 在小样本的情况，稀疏的逆协方差估计往往比收缩的协方差估计更好。在相反的情况下，或者对于非常相关的数据，它们可能在数值上不稳定。

# GraphLasso 估计器使用 L1 惩罚确立精度矩阵的稀疏性： alpha 参数越高，精度矩阵的稀疏性越大。
# 相应的 GraphLassoCV 对象使用交叉验证来自动设置 alpha 参数。


# 示例： 协方差矩阵和 精度矩阵基于最大似然度估计,收缩估计和稀疏估计的比较_
"""
使用GraphicalAllasso估计器从少量样本中学习协方差和稀疏精度。
为了估计概率模型（例如高斯模型），估计精度矩阵，即逆协方差矩阵，与估计协方差矩阵一样重要。事实上，高斯模型由精度矩阵参数化。
为了在有利的恢复条件下，我们从具有稀疏逆协方差矩阵的模型中采样数据。
此外，我们确保数据没有太多的相关性（限制精度矩阵的最大系数），并且精度矩阵中没有无法恢复的小系数。
此外，对于少量的观测，恢复相关矩阵比恢复协方差更容易，因此我们可以缩放时间序列。

这里，样本数量略大于维度数量，因此经验协方差仍然是可逆的。
然而，由于观测值具有强相关性，经验协方差矩阵是病态的，因此其逆矩阵——经验精度矩阵——与基本事实相差甚远。

如果我们使用l2收缩，就像使用Ledoit-Wolf估计一样，由于样本数很小，我们需要收缩很多。
因此，Ledoit-Wolf精度相当接近地面真值精度，也就是说离对角线不远，但失去了非对角线结构。

l1惩罚估计可以恢复部分非对角结构。它学习的精度很低。它无法恢复精确的稀疏模式：它检测到太多的非零系数。
然而，估计的l1的最高非零系数对应于基本真值中的非零系数。最后，l1精度估计的系数偏向于零：由于惩罚，它们都小于相应的地面真值，如图所示。

请注意，调整精度矩阵的颜色范围以提高图形的可读性。不显示经验精度的完整值范围。

GraphicalAsso设置模型稀疏性的alpha参数由GraphicalAssoCV中的内部交叉验证设置。
如图2所示，计算交叉验证分数的网格在最大值附近迭代细化。


"""
import numpy as np
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
import matplotlib.pyplot as plt

# #############################################################################
# Generate the data
n_samples = 60
n_features = 20

prng = np.random.RandomState(1)
# 生成稀疏对称正定矩阵
prec = make_sparse_spd_matrix(n_features, alpha=.98, smallest_coef=.4, largest_coef=.7, random_state=prng)
cov = linalg.inv(prec)
d = np.sqrt(np.diag(cov))
cov /= d
cov /= d[:, np.newaxis]
prec *= d
prec *= d[:, np.newaxis]
X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
X -= X.mean(axis=0)
X /= X.std(axis=0)

# #############################################################################
# Estimate the covariance
emp_cov = np.dot(X.T, X) / n_samples

model = GraphicalLassoCV()
model.fit(X)
cov_ = model.covariance_
prec_ = model.precision_

lw_cov_, _ = ledoit_wolf(X)
lw_prec_ = linalg.inv(lw_cov_)

# #############################################################################
# Plot the results
plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.02, right=0.98)

# plot the covariances
covs = [
    ('Empirical', emp_cov),
    ('Ledoit-Wolf', lw_cov_),
    ('GraphicalLassoCV', cov_),
    ('True', cov)
]
vmax = cov_.max()
for i, (name, this_cov) in enumerate(covs):
    plt.subplot(2, 4, i + 1)
    plt.imshow(this_cov, interpolation='nearest', vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.title('%s covariance' % name)

# plot the precisions
precs = [
    ('Empirical', linalg.inv(emp_cov)),
    ('Ledoit-Wolf', lw_prec_),
    ('GraphicalLasso', prec_),
    ('True', prec)
]
vmax = .9 * prec_.max()
for i, (name, this_prec) in enumerate(precs):
    ax = plt.subplot(2, 4, i + 5)
    plt.imshow(np.ma.masked_equal(this_prec, 0), interpolation='nearest', vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.title('%s precision' % name)
    if hasattr(ax, 'set_facecolor'):
        ax.set_facecolor('.7')
    else:
        ax.set_axis_bgcolor('.7')

# plot the model selection metric
plt.figure(figsize=(4, 3))
plt.axes([.2, .15, .75, .7])
plt.plot(model.cv_results_["alphas"], model.cv_results_["mean_score"], 'o-')
plt.axvline(model.alpha_, color='.5')
plt.title('Model selection')
plt.ylabel('Cross-validation score')
plt.xlabel('alpha')

plt.show()
