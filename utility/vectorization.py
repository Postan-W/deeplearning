"""
向量化工具
下面用一个非常简洁例子展示数据向量化为one-hot。假设review[[]]这个列表保存的是观众对一个电影的评论,每个子元素是一条
完整的评论，评论里的每一个词已经被分割好，并用对应的数字表示了"""
import numpy as np

def vectorization(sequnences,dimension=10):
    vector = np.zeros((len(sequnences),dimension))
    for i,review in enumerate(sequnences):
        vector[i,review] = 1
        return vector

"""
Output:
[[0. 1. 1. 1. 1. 1. 1. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
"""