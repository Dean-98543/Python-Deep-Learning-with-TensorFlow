# Author: Dean
# Email: yangmingnjau@163.com
# Date: 2019/12/22 10:11
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from numpy.linalg import eig

# 1. 载入数据集
iris = load_iris()
print(dir(iris))        # ['DESCR', 'data', 'feature_names', 'filename', 'target', 'target_names']
X = iris.data
print(X.shape)      # (150, 4) 所有的数据都在data中，一共有150个数据点(150行)，每个数据点有四个特征值(4列)


X = X - X.mean(axis=0)
# numpy.mean(a, axis=None, dtype=None, out=None, keepdims=False) axis=0时表示纵轴平均，输出的是格式（1，x）的格式

# 2. 计算协方差矩阵
X_cov = np.cov(X.T, ddof=0)
# 函数原型def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None,aweights=None)
# m: 一维或则二维的数组，默认情况下每一行代表一个变量（属性），每一列代表一个观测
# y: 与m具有一样的形式的一组数据
# rowvar: 默认为True,此时每一行代表一个变量（属性），每一列代表一个观测；为False时，则反之
# bias: 默认为False,此时标准化时除以n-1；反之为n。其中n为观测数
# ddof: 类型是int，当其值非None时，bias参数作用将失效。当ddof=1时，将会返回无偏估计（除以n-1），即使指定了fweights和aweights参数；当ddof=0时，则返回简单平均值。
# frequency weights: 一维数组，代表每个观测要重复的次数（相当于给观测赋予权重）
# analytic weights: 一维数组，代表观测矢量权重。对于被认为“重要”的观察,这些相对权重通常很大,而对于被认为不太重要的观察,这些相对权重较小。如果ddof = 0,则可以使用权重数组将概率分配给观测向量。
print(X_cov)
# [[ 0.68112222 -0.04215111  1.26582     0.51282889]
#  [-0.04215111  0.18871289 -0.32745867 -0.12082844]
#  [ 1.26582    -0.32745867  3.09550267  1.286972  ]
#  [ 0.51282889 -0.12082844  1.286972    0.57713289]]

# 3. 计算协方差矩阵的特征值和特征向量
eigenvalues, eigenvectors = eig(X_cov)
print(eigenvalues)
# [4.20005343 0.24105294 0.0776881  0.02367619]
print(eigenvectors)
# [[ 0.36138659 -0.65658877 -0.58202985  0.31548719]
#  [-0.08452251 -0.73016143  0.59791083 -0.3197231 ]
#  [ 0.85667061  0.17337266  0.07623608 -0.47983899]
#  [ 0.3582892   0.07548102  0.54583143  0.75365743]]

# 4. 计算各特征值的贡献率
tot = sum(eigenvalues)
var_exp = [(i / tot) for i in sorted(eigenvalues, reverse=True)]
print(sorted(eigenvalues, reverse=True))        # sorted() 函数对所有可迭代的对象进行排序操作
# [4.2000534279946375, 0.24105294294244267, 0.07768810337596671, 0.0236761923536266]
print(var_exp)      # [0.924618723201727, 0.053066483117067784, 0.017102609807929766, 0.005212183873275401]

cum_var_exp = np.cumsum(var_exp)
# numpy.cumsum(a,  axis=None, dtype=None, out=None)
# 按照所给定的轴参数返回元素的梯形累计和，axis=0，按照行累加。axis=1，按照列累加。axis不给定具体值，就把numpy数组当成一个一维数组
print(cum_var_exp)      # [0.92461872 0.97768521 0.99478782 1.        ]

# 5. 可视化效果
plt.bar(range(1, 5), var_exp, alpha=0.5, align='center', label='individual var')
plt.step(range(1, 5), cum_var_exp, where='mid', label='cumulative var')
plt.ylabel('variance rtion')
plt.xlabel('principal components')
plt.legend(loc='best')
plt.show()

