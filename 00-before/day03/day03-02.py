import matplotlib.pyplot as plt

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

"""
数字数据集由8x8像素的数字图像组成。
数据集的images属性为每个图像存储8x8个灰度值数组。我们将使用这些数组来可视化前4幅图像。
数据集的target属性存储每个图像所代表的数字，这包括在下面4个绘图的标题中。

如果使用的是图像文件（例如“png”文件），则使用matplotlib.pyplot.imread加载。
"""

digits = datasets.load_digits()

# # 1. 展示
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     # ax.set_axis_off()  # 坐标
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title('Training: %i' % label)

# 2. 分类
# 要对这些数据使用分类器，我们需要将图像展平
# 将每个二维灰度值数组从形状（8，8）转换为形状（64，）。
# 随后，整个数据集将是形状（n_samples, n_features），
# 其中n_samples是图像的数量，n_features是每个图像中的像素总数。

# flatten the images  展平图像
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets  将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

# Learn the digits on the train subset
clf.fit(X_train, y_train)  # 训练训练集

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)  # 预测测试集
_, axes = plt.subplots(nrows=1, ncols=10, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}')

print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")  # 输出分类报告

# plt.show()

disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
# 绘制真实数字值和预测数字值的混淆矩阵。

plt.show()


