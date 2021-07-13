# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/7/13 10:46
# @Function: 示例3 ： Mauna Loa CO2 数据中的 GRR
# 

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared


def load_mauna_loa_atmospheric_co2():
    ml_data = fetch_openml(data_id=41187, as_frame=False)
    months = []
    ppmv_sums = []
    counts = []

    y = ml_data.data[:, 0]
    m = ml_data.data[:, 1]
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            # aggregate monthly sum to produce average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months).reshape(-1, 1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts
    return months, avg_ppmvs


X, y = load_mauna_loa_atmospheric_co2()
#
# Kernel with parameters given in GPML book
k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
k2 = 2.4**2 * RBF(length_scale=90.0) * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
# medium term irregularity
k3 = 0.66**2 * RationalQuadratic(length_scale=1.2, alpha=0.78)
k4 = 0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19**2)  # noise terms
kernel_gpml = k1 + k2 + k3 + k4

gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0, optimizer=None, normalize_y=True)
gp.fit(X, y)

print("GPML kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f" % gp.log_marginal_likelihood(gp.kernel_.theta))
#
# Kernel with optimized parameters
k1 = 50.0**2 * RBF(length_scale=50.0)  # RBF解释一个长期平稳的上升趋势
k2 = 2.0**2 * RBF(length_scale=100.0) \
     * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")  # 季节性因素。固定周期为1年
# RationalQuadratic解释较小的中期不规则性
k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
# WhiteKernel处理噪声
k4 = 0.1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-5, np.inf))  # noise terms
kernel = k1 + k2 + k3 + k4
# kernel = k1 + k3 + k4  # 可以多测试几种

gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True)
gp.fit(X, y)

print("\nLearned kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f" % gp.log_marginal_likelihood(gp.kernel_.theta))

X_ = np.linspace(X.min(), X.max() + 30, 1000)[:, np.newaxis]
y_pred, y_std = gp.predict(X_, return_std=True)

plt.scatter(X, y, c='k')
plt.plot(X_, y_pred)
plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std, alpha=0.5, color='k')
plt.xlim(X_.min(), X_.max())
plt.xlabel("Year")
plt.ylabel(r"CO$_2$ in ppm")
plt.title(r"Atmospheric CO$_2$ concentration at Mauna Loa")
plt.tight_layout()
plt.show()


"""
Learned kernel: 
    2.62**2 * RBF(length_scale=51.6) +  
    0.155**2 * RBF(length_scale=91.4) * ExpSineSquared(length_scale=1.48, periodicity=1) + 
    0.0315**2 * RationalQuadratic(alpha=2.88, length_scale=0.968) + 
    0.011**2 * RBF(length_scale=0.122) + WhiteKernel(noise_level=0.000126)

GPML kernel: 
    66**2 * RBF(length_scale=67) + 
    2.4**2 * RBF(length_scale=90) * ExpSineSquared(length_scale=1.3, periodicity=1) + 
    0.66**2 * RationalQuadratic(alpha=0.78, length_scale=1.2) + 
    0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.0361)


"""
