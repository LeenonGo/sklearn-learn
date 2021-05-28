"""
学习方法：
    1. 复杂或多种图形显示在同一图像上的话，建议一个图形一个图形的显示再分析，就会清楚很多；
    2. 编程基础或理解能力较弱可以全篇打上注释，再一行一行的看，可以输出对比。

1. np.random.seed(0)  设置相同的seed，每次生成的随机数相同

2. data[a:b, n:m]  取data第a到b中的第n到m个属性，  即二维变量，取行和列。X[:, np.newaxis]增加一个维度。作用：将一维的数据转变成一个矩阵，与代码后面的权重矩阵进行相乘

3. coef 预测系数，先理解为参数权重；intercept_ 补偿值

4. np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等；
   np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。

5. 生成指定数
    np.logspace(start, end, num, base) base为底数  np.logspace(-10, -2, 10， base)表示10的-10次方到10的-2次方的等比数列，取10个数
    np.linspace(start, end, num) 均匀取值  与arrange类似：arrange取步长，linespace取个数
    np.random.permutation(x)   随机排序
    np.meshgrid(x, y)  生成x y的坐标矩阵，即按x和y将图形分为网格

6. 岭回归：
    解决特征数量比样本量多的问题
    缩减算法,判断哪些特征重要或者不重要，有点类似于降维的效果
    缩减算法可以看作是对一个模型增加偏差的同时减少方差

7. 划线
    plot   单条线
    scatter(x, y)   散点图
    axhline: 平行于X轴的水平参考线
        plt.axhline(y=0.0, c="r", ls="--", lw=2)
        y：水平参考线的出发点  c：参考线的线条颜色   ls：参考线的线条风格   lw：参考线的线条宽度

8. 捕获拟合参数噪声使得模型不能归纳新的数据称为过拟合。岭回归产生的偏差被称为 正则化。正则化是为了防止过拟合。

9. 把一些系数设为0。这些方法称为 稀疏法，稀疏可以看作是奥卡姆剃刀的应用：模型越简单越好。

10. scikit-learn 里 Lasso 对象使用 coordinate descent（坐标下降法） 方法解决 lasso 回归问题，对于大型数据集很有效

11. ravel()、flatten()、squeeze()  将多维转为一维，对多维数据进行扁平化操作
    ravel()：如果没有必要，不会产生源数据的副本
    flatten()：返回源数据的副本。返回新对象
    squeeze()：只能对维数为1的维度降维

12. expit函数，也称为logistic sigmoid函数，定义为expit（x）= 1 /（1 + exp（-x））。 它是logit函数的反函数。


01-02：     属性的分布图显示（2维/3维）
01-02-2：  使用K近邻分类器  数据集分解为训练集和测试集
01-03：    使用线性回归预测疾病级别
01-03-1：  线性回归和岭回归对比：如果每个维度的数据点很少，观察噪声就会导致很大的方差
01-03-2a： 对岭回归进行简单学习：岭参数 alpha 越大，偏差越大，方差越小。解决特征数量比样本量多的问题
01-04：    使用线性回归查看特征与目标的分布
01-04-2：  Lasso回归模型
01-05：    逻辑回归使用逻辑曲线将值分类为0或1：对于分类问题，线性回归不是一个好的方法，使用逻辑回归
01-05-2：  使用逻辑回归对多维特征进行训练及预测


线性回归适用于连续型  逻辑回归适用于离散型
"""

