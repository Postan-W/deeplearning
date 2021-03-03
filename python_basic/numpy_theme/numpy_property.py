"""
很多时候可以声明 axis。axis=0，表示沿着第 0 轴进行操作，即对每一列进行操作；axis=1，表示沿着第1轴进行操作，即对每一行进行操作。
NumPy 的数组中比较重要 ndarray 对象属性有：
属性	说明
ndarray.ndim	秩，即轴的数量或维度的数量
ndarray.shape	数组的维度，对于矩阵，n 行 m 列
ndarray.size	数组元素的总个数，相当于 .shape 中 n*m 的值
ndarray.dtype	ndarray 对象的元素类型
ndarray.itemsize	ndarray 对象中每个元素的大小，以字节为单位
ndarray.flags	ndarray 对象的内存信息
ndarray.real	ndarray元素的实部
ndarray.imag	ndarray 元素的虚部
ndarray.data	包含实际数组元素的缓冲区，由于一般通过数组的索引获取元素，所以通常不需要使用这个属性。
ndarray.ndim
"""

import numpy as np

a = np.arange(16)
print(a,a.ndim)  # a 现只有一个维度
# 现在调整其大小
b = a.reshape(2, 4, 2)  # b 现在拥有三个维度
print(b,b.ndim)

a = np.array([[1,2,3],[4,5,6]])
print (a.shape)

#调整数组形状
a = np.array([[1,2,3],[4,5,6]])
a.shape =  (3,2)#效果与reshape函数一样,但是reshape返回了一个对象(或者叫引用),不过每个具体的元素a和b是共用的
print (a)
b = a.reshape([6,])
print(b)
print(a.shape)
print(a)#由此可见a是按照shape来输出或被使用的


# 数组的 dtype 为 int8（一个字节）
x = np.array([1, 2, 3, 4, 5], dtype=np.int8)
print(x.itemsize)

#ndarray.itemsize 以字节的形式返回数组中每一个元素的大小
# 数组的 dtype 现在为 float64（八个字节）
y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
print(y.itemsize)

"""
ndarray.flags
ndarray.flags 返回 ndarray 对象的内存信息，包含以下属性：

属性	描述
C_CONTIGUOUS (C)	数据是在一个单一的C风格的连续段中
F_CONTIGUOUS (F)	数据是在一个单一的Fortran风格的连续段中
OWNDATA (O)	数组拥有它所使用的内存或从另一个对象中借用它
WRITEABLE (W)	数据区域可以被写入，将该值设置为 False，则数据为只读
ALIGNED (A)	数据和所有元素都适当地对齐到硬件上
UPDATEIFCOPY (U)	这个数组是其它数组的一个副本，当这个数组被释放时，原数组的内容将被更新

"""
x = np.array([1,2,3,4,5])
print (x.flags)
