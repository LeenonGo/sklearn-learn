# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/26 15:20
# @Function: 变分贝叶斯高斯混合  https://www.scikitlearn.com.cn/0.21.3/20/#212
# BayesianGaussianMixture 对象实现了具有变分推理算法的高斯混合模型的变体

# 估计算法: 变分推断（variational inference）  https://www.scikitlearn.com.cn/0.21.3/20/#2121-variational-inference
# 变分推断是期望最大化（EM）的扩展，它最大化了模型证据（包括先验）的下界，而不是数据的似然性。
# 变分方法的原理与期望最大化相同（二者都是迭代算法，在寻找由每种模型的混合所产生的每个点的概率和根据所分配的点拟合之间两步交替），
# 但是变分方法通过整合先验分布信息来增加正则化限制。
# 这避免了期望最大化解决方案中常出现的奇异性，但是也给模型带来了微小的偏差。变分的计算过程通常比较慢，但不会慢到无法使用。

# 由于它的贝叶斯特性，变分算法比预期最大化（EM）需要更多的超参数（即先验分布中的参数），
# 其中最重要的就是浓度参数 weight_concentration_prior 。
# 指定一个低浓度先验，将会使模型将大部分的权重分配到少数分量上，而其余分量的权重则趋近 0。
# 而高浓度先验将会使混合模型中的更多的分量在混合模型中都有相当比列的权重。

# BayesianGaussianMixture 类的参数实现提出了两种权重分布先验：
#   一种是利用 Dirichlet distribution（狄利克雷分布）的有限混合模型
#   另一种是利用 Dirichlet Process（狄利克雷过程）的无限混合模型。
# 在实际应用中，狄利克雷过程推理算法是近似的，并且使用具有固定最大分量数的截断分布（称之为 Stick-breaking representation）,
# 而其使用的分量数实际上总是依赖数据。

#
# 示例：变分贝叶斯-高斯混合模型的浓度先验型分析
# BayesianGaussianMixture类可以自动调整其混合组分的数量。参数weight_concentration_prior与具有非零权重的组件的结果数量有直接联系。
# 为之前的浓度指定一个较低的值将使模型将大部分权重放在少数组分上，并将其余组分的权重设置为非常接近于零。之前的浓度值越高，混合物中的活性组分就越多。
# Dirichlet Process 允许预先定义无限个组件，并自动选择正确的组件数：它仅在必要时激活组件。
# 相反，具有Dirichlet分布先验的经典有限混合模型将倾向于更均匀的加权成分，因此倾向于将自然簇划分为不必要的子成分。

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.mixture import BayesianGaussianMixture


def plot_ellipses(ax, weights, means, covars):
    for n in range(means.shape[0]):
        eig_vals, eig_vecs = np.linalg.eigh(covars[n])
        unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
        angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
        # Ellipse needs degrees
        angle = 180 * angle / np.pi
        # eigenvector normalization
        eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
        ell = mpl.patches.Ellipse(means[n], eig_vals[0], eig_vals[1],
                                  180 + angle, edgecolor='black')
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(weights[n])
        ell.set_facecolor('#56B4E9')
        ax.add_artist(ell)


def plot_results(ax1, ax2, estimator, X, y, title, plot_title=False):
    ax1.set_title(title)
    ax1.scatter(X[:, 0], X[:, 1], s=5, marker='o', color=colors[y], alpha=0.8)
    ax1.set_xlim(-2., 2.)
    ax1.set_ylim(-3., 3.)
    ax1.set_xticks(())
    ax1.set_yticks(())
    plot_ellipses(ax1, estimator.weights_, estimator.means_,
                  estimator.covariances_)

    ax2.get_xaxis().set_tick_params(direction='out')
    ax2.yaxis.grid(True, alpha=0.7)
    for k, w in enumerate(estimator.weights_):
        ax2.bar(k, w, width=0.9, color='#56B4E9', zorder=3,
                align='center', edgecolor='black')
        ax2.text(k, w + 0.007, "%.1f%%" % (w * 100.),
                 horizontalalignment='center')
    ax2.set_xlim(-.6, 2 * n_components - .4)
    ax2.set_ylim(0., 1.1)
    ax2.tick_params(axis='y', which='both', left=False,
                    right=False, labelleft=False)
    ax2.tick_params(axis='x', which='both', top=False)

    if plot_title:
        ax1.set_ylabel('Estimated Mixtures')
        ax2.set_ylabel('Weight of each component')


# Parameters of the dataset
random_state, n_components, n_features = 2, 3, 2
colors = np.array(['#0072B2', '#F0E442', '#D55E00'])

covars = np.array([[[.7, .0], [.0, .1]],
                   [[.5, .0], [.0, .1]],
                   [[.5, .0], [.0, .1]]])
samples = np.array([200, 500, 200])
means = np.array([[.0, -.70],
                  [.0, .0],
                  [.0, .70]])

# mean_precision_prior= 0.8 to minimize the influence of the prior
estimators = [
    ("Finite mixture with a Dirichlet distribution\nprior and "
     r"$\gamma_0=$", BayesianGaussianMixture(
        weight_concentration_prior_type="dirichlet_distribution",
        n_components=2 * n_components, reg_covar=0, init_params='random',
        max_iter=1500, mean_precision_prior=.8,
        random_state=random_state), [0.001, 1, 1000]),
    ("Infinite mixture with a Dirichlet process\n prior and" r"$\gamma_0=$",
     BayesianGaussianMixture(
         weight_concentration_prior_type="dirichlet_process",
         n_components=2 * n_components, reg_covar=0, init_params='random',
         max_iter=1500, mean_precision_prior=.8,
         random_state=random_state), [1, 1000, 100000])
]

# Generate data
rng = np.random.RandomState(random_state)
X = np.vstack([
    rng.multivariate_normal(means[j], covars[j], samples[j])
    for j in range(n_components)])
y = np.concatenate([np.full(samples[j], j, dtype=int)
                    for j in range(n_components)])

# Plot results in two different figures
for (title, estimator, concentrations_prior) in estimators:
    plt.figure(figsize=(4.7 * 3, 8))
    plt.subplots_adjust(bottom=.04, top=0.90, hspace=.05, wspace=.05,
                        left=.03, right=.99)

    gs = gridspec.GridSpec(3, len(concentrations_prior))
    for k, concentration in enumerate(concentrations_prior):
        estimator.weight_concentration_prior = concentration
        estimator.fit(X)
        plot_results(plt.subplot(gs[0:2, k]), plt.subplot(gs[2, k]), estimator,
                     X, y, r"%s$%.1e$" % (title, concentration),
                     plot_title=k == 0)

plt.show()
