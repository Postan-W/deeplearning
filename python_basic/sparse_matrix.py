#用字典存储稀疏矩阵

import numpy as np
sparse_matrix = np.array([[54, 33, 58, 27 ,72],
 [75 ,12 ,71 ,10, 59],
 [35, 78  ,7 ,91, 26],
 [43 ,27, 22, 96, 75],
 [ 7, 55,15 ,41 , 1]])
dict_matrix = {}
#把它稀疏化。我们可以是每个元素减去70，小于0的置为0
for i in range(sparse_matrix.shape[0]):
    for j in range(sparse_matrix.shape[1]):
        if sparse_matrix[i,j] <= 70:
            sparse_matrix[i,j] = 0
        else:
            dict_matrix["(%d,%d)" % (i, j)] = sparse_matrix[i,j]

#保存形状
dict_matrix["shape"] = sparse_matrix.shape

#下面获取一个用字典保存的矩阵
shape = dict_matrix['shape']
steps = 0
for i in range(shape[0]):
    for j in range(shape[1]):
        steps += 1
        if '(%d,%d)'%(i,j) in dict_matrix.keys():
            if steps%shape[0] == 0:
                print(dict_matrix['(%d,%d)' % (i, j)])
            else:
                print(dict_matrix['(%d,%d)' % (i, j)], end=" ")
        else:
            if steps%shape[0] == 0:
                print(0)
            else:
                print(0,end=" ")