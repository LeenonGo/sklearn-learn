# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/6/8 14:23
# @Function:
# 

"""
基于特征脸算法和支持向量机的人脸识别实例
数据集地址：  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
"""
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # 日志打印

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
# 下载数据集，每人至少min_faces_per_person张脸。resize用于调整每个人脸图片大小的比率，默认0.5。
# 可以先将数据集下载至本地：C:\Users\Lee\scikit_learn_data\lfw_home

n_samples, h, w = lfw_people.images.shape  # # 1288 50 37

X = lfw_people.data
n_features = X.shape[1]  # 1850

y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

# print("Total dataset size:")
# print("n_samples: %d" % n_samples)  # 1288
# print("n_features: %d" % n_features)  # 1850
# print("n_classes: %d" % n_classes) # 7 --> 只有7个人有70以上人脸min_faces_per_person

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)  # 划分训练集和测试集

n_components = 150

print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")

t0 = time()
X_train_pca = pca.transform(X_train)  # 降至150个特征
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=5, iid=False)
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=5)  # 和 day06-01.py一样
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))  # 输出混淆矩阵


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=4, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=.01, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()


"""
该例子
加载数据后，先对数据进行降维。设置n_components = 150
然后使用GridSearchCV训练模型，并找出最优参数
y_pred = clf.predict(X_test_pca)，运行测试集
得出两个图：
    1. 预测结果X_test
    2. 特征eigenfaces，可以理解为降维之后的数据，即原数据是由1850个特征向量构成的，输出的图像为150个特征向量构成的，看起来是“模糊的”

"""


