#使用预训练好的词嵌入
"""
没有足够的数据来学习真正强大的特征，那么使用预训练的词向量表示，但是前提是学习到的特征是非常通用的，最好是抓住了
语言结构的一般特点，使得它的适应性更强。比如是从大量的科幻小说
语料中学习到的词向量，那么再将训练好的词向量运用到科幻小说文本处理相应的任务中去则应该是合理的，在其他文学类型
的任务中可能会有一些作用，但也可能有些词向量没什么意义。
"""
"""
需要构建一个可以加载到Embedding层中的嵌入矩阵。它必须是一个形状为(max_words,  embedding_dim)的矩阵，
对于单词索引（在分词时构建）中索引为i的单词，这个矩阵的元素i就是这个单词对应的embedding_dim维向量
"""
import numpy as np
import pickle
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

glove_dir = "..\\glove\\glove.6B.100d.txt"
embeddings_index = {}
with open(glove_dir,'r',encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
with open('..\\glove\\glove.6B.100d_dict.pk','wb') as f:
    pickle.dump(embeddings_index,f)

#生成词嵌入矩阵
embedding_dim = 100
#只考虑1000个词的嵌入
max_words = 1000
#句子长度为100个词
maxlen = 100
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
word_index = reuters.get_word_index()
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        #上面返回词向量的操作意味着embeddings_index中没有的词就返回0向量了
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
"""将 (样本数也就是句子数，句子长度maxlen,嵌入维度embedding_dim)这样的3维张量展平为(样本数，句子)也就是(samples,maxlen*embedding_dim)的2维张量形式输入密集连接层"""
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
"""
需要冻结Embedding层（即将其trainable属性设为False），其原理和预训练的卷积神经网络特征相同，你已经很熟悉了。
如果一个模型的一部分是经过预训练的（如Embedding层），而另一部分是随机初始化的（如分类器），
那么在训练期间不应该更新预训练的部分是的，以避免丢失它们所保存的信息。随机初始化的层会引起较大的梯度更新，
会破坏已经学到的特征。
"""
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(train_data,train_labels, epochs=10,batch_size=32,validation_data=(test_data, test_labels))
model.save_weights('..\\records\\models\\pre_trained_glove_model.h5')

