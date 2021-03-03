import numpy as np
#生成网格点，网格点返回网格点的横纵坐标矩阵
x,y = np.meshgrid(np.array([1,2,3,4]),np.array([1,4,9,16,25]))
print(x)
print(y)
