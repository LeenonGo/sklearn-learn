from sklearn import datasets

import matplotlib.pyplot as plt

"""
这个数据集包含150个样本，每个样本包含4个特征：花萼长度，花萼宽度，花瓣长度，花瓣宽度，详细数据可以通过``iris.DESCR``查看。
如果原始数据不是``(n_samples, n_features)``的形状时，使用之前需要进行预处理以供scikit-learn使用。
"""
iris = datasets.load_iris()
data = iris.data
# data.shape(150, 4)

"""
数据预处理样例:digits数据集(手写数字数据集)
"""
digits = datasets.load_digits()
# print(digits.images.shape)  # (1797, 8, 8)  数据集包含1797个手写数字的图像，每个图像为8*8像素
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)
data = digits.images.reshape((digits.images.shape[0], -1))
# estimator = Estimator(param1=1, param2=2)  # Estimator是TensorFlow的高层API
# estimator.fit(data)


