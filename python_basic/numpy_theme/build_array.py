#numpy.empty(shape, dtype = float, order = 'C'),order	有"C"和"F"两个选项,分别代表，行优先和列优先，在计算机内存中的存储元素的顺序。

import numpy as np
x = np.empty([3,2], dtype = int)
print (x)

# 默认为浮点数
x = np.zeros(5)
print(x)

# 设置类型为整数
y = np.zeros((5,), dtype=np.int)
print(y)

# 自定义类型
z = np.zeros((2, 2), dtype=[('x', 'i4'), ('y', 'i4')])
print(z)

x = np.ones(5)
print(x)

# 自定义类型
x = np.ones([2, 2], dtype=int)
print(x)

x = np.ones(5)
print(x)

# 自定义类型
x = np.ones([2, 2], dtype=int)
print(x)

#asarray
"""
参数	    描述
a	    任意形式的输入参数，可以是，列表, 列表的元组, 元组, 元组的元组, 元组的列表，多维数组
dtype	数据类型，可选
order	可选，有"C"和"F"两个选项,分别代表，行优先和列优先，在计算机内存中的存储元素的顺序
"""
x =  [1,2,3]
a = np.asarray(x)
print (a)

x =  [(1,2,3),(4,5)]
a = np.asarray(x)
print (a)

x =  [1,2,3]
a = np.asarray(x, dtype =  float)
print (a)


#numpy.frombuffer(buffer, dtype = float, count = -1, offset = 0)
"""
参数	描述
buffer	可以是任意对象，会以流的形式读入。
dtype	返回数组的数据类型，可选
count	读取的数据数量，默认为-1，读取所有数据。
offset	读取的起始位置，默认为0。
"""
s =  b'Hello World'#python3中的字节格式必须加上b前缀表明其是bytes类型
a = np.frombuffer(s, dtype =  'S1')
print (a)

#numpy.fromiter(iterable, dtype, count=-1)
"""
numpy.fromiter 方法从可迭代对象中建立 ndarray 对象，返回一维数组
参数	描述
iterable	可迭代对象
dtype	返回数组的数据类型
count	读取的数据数量，默认为-1，读取所有数据
"""
a = range(10)
print(type(a))
a = list(a)
x=np.fromiter(a,dtype=float)
print(x)

#numpy.arange(start, stop, step, dtype)
x = np.arange(10,20,2,dtype='int64')
print (x)
print(x.itemsize)

#linespace与arange效果是一样的，只是用法上有些许差异
#np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)

"""
numpy.linspace 函数用于创建一个一维数组，数组是一个等差数列构成的
参数	描述
start	序列的起始值
stop	序列的终止值，如果endpoint为true，该值包含于数列中
num	要生成的等步长的样本数量，默认为50
endpoint	该值为 true 时，数列中包含stop值，反之不包含，默认是True。
retstep	如果为 True 时，生成的数组中会显示间距，反之不显示。
dtype	ndarray 的数据类型
"""

a = np.linspace(1,50,50,endpoint=True,retstep=True,dtype='int32')
print(a)


#np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
"""
numpy.logspace 函数用于创建一个于等比数列
参数	描述
start	序列的起始值为：base ** start
stop	序列的终止值为：base ** stop。如果endpoint为true，该值包含于数列中
num	要生成的等步长的样本数量，默认为50
endpoint	该值为 true 时，数列中中包含stop值，反之不包含，默认是True。
base	对数 log 的底数。
dtype	ndarray 的数据类型

"""
a = np.logspace(0,9,10,base=2)
print (a)
