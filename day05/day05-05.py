# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/6/7 17:29
# @Function: 分解: 将一个信号转换成多个成份并且加载 -- 独立成分分析: ICA
# 

# 独立成分分析（ICA） 可以提取数据信息中的独立成分，
# 这些成分载荷的分布包含了最多的独立信息。
# 该方法能够恢复 non-Gaussian（非高斯） 独立信号

"""
从噪声数据估计源的一个例子。
独立分量分析（ICA）用于估计给定噪声测量的源。
想象3种乐器同时演奏，3个麦克风记录混合信号。ICA用于恢复源，即每个乐器所演奏的音乐。
重要的是，主成分分析无法恢复我们的仪器，因为相关的信号反映了非高斯过程。
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import FastICA, PCA

# #############################################################################
# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal  信号1：正弦信号
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal  信号2：方形信号
# np.sign: X>0时，取1； X=0时，取0； X<0时，取-1
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal  信号3：锯齿信号

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations 乘法

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
print(np.allclose(X,  np.dot(S_, A_.T) + ica.mean_))
# 通过还原分解来“证明”ICA模型是适用的。
# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# #############################################################################
# Plot results

plt.figure()

models = [X, S, S_, H]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)


plt.tight_layout()
plt.show()







