import numpy as np
#将两个等长数组的对应元素组成couple作为一个子数组
a= np.c_[np.array([1,2,3,4]), np.array([5,6,7,8])]
print(a)
print(type(a))
print(a.shape)