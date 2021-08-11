# -*- coding: utf-8 -*-
# @Author  : Lee
# @Time    : 2021/8/11 21:33
# @Function: 3.4. 模型持久化  https://www.scikitlearn.com.cn/0.21.3/33/
#
#
import pickle
# s = pickle.dumps(clf)  # 持久化
# clf2 = pickle.loads(s)  # 加载

# pickle（和通过扩展的 joblib），在安全性和可维护性方面存在一些问题。 有以下原因，
#   绝对不要使用未经 pickle 的不受信任的数据，因为它可能会在加载时执行恶意代码。
#   虽然一个版本的 scikit-learn 模型可以在其他版本中加载，但这完全不建议并且也是不可取的。
#   还应该了解到，对于这些数据执行的操作可能会产生不同及意想不到的结果。
# 

