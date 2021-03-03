import numpy as np

a = np.array([[1,2],[3,4]])
a.ravel()[1] = 100
print(a) #ravel()返回的对象与原对象共用内存，只是子元素的排列改变了而已
a.flatten()[1] = 999
print(a) #返回复制的内容