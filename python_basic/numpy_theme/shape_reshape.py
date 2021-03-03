#shape是个属性，作用是显示或改变数组形状。reshape是个函数，返回一个对象，该对象和调用者共用元素的内存
import numpy as np
a = np.zeros((2,3),dtype="int64")
b = np.zeros([3,2],dtype="int64")
print(a,b)
print(a.shape)
a.shape = [6,]
print(a)

d= b.reshape([6,])
print(b)
print(d)
"""outputs:[[0 0]
 [0 0]
 [0 0]]
 
 [0 0 0 0 0 0]
 """
#由此可见reshape函数并不是改变调用者本身的形状，而是返回一个使用调用者数据，并且形状为函数参数定义的形状的对象
#下面证明reshape返回的对象和调用者共用元素内存，换句话说共用具体的元素。
b[2] = 9999
print(b)
print(d)
"""
[[   0    0]
 [   0    0]
 [9999 9999]]
 
[   0    0    0    0 9999 9999]
"""