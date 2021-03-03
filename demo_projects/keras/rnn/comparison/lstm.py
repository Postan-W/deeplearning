"""
这个项目用lstm来实现文本分类。glove词向量文件太大了，这里引用的是项目范围外的glove文件，移植会出错。
Embedding层的输入是(samples,sequence_length),输出是(samples,sequence_length,embedding_dimensionality),这个很好理解。
"""
from keras.layers import Embedding
