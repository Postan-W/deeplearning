import numpy as np

a = np.arange(6).reshape(2,3)
print ('原始数组是：')
print (a)
print ('以 C 风格顺序排序：')
for x in np.nditer(a, order =  'C'):
    print (x, end=", " )
print ('\n')
print ('以 F 风格顺序排序：')
for x in np.nditer(a, order =  'F'):
    print (x, end=", " )

#numpy.ndarray.flat 是一个数组元素迭代器
a = np.arange(9).reshape(3, 3)
print('原始数组：')
for i in a:
    print(i)

# 对数组中每个元素都进行处理，可以使用flat属性，该属性是一个数组元素迭代器：
print('迭代后的数组：')
for element in a.flat:
    
    print(element)