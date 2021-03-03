
import string
import numpy as np
samples = ['concatenate is the operation of combining several vectors into one','China is a country with 56 nationnalities']

#单词级ont-hot编码。
#字典序号从零开始
token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index)

#设定句子的最大长度
max_length = 8
#初始化
results = np.zeros(shape=(len(samples),max_length,max(token_index.values())+1))

#构造
for i,sample in enumerate(samples):
    for j,word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            results[i,j,index] = 1

print(results)



#字符级one-hot编码
#所有可打印的ASCII字符
characters  =string.printable
token_index2 = dict(zip(characters,range(0,len(characters))))
print(token_index2)
max_length2 = 30
results2 = np.zeros((len(samples),max_length2,max(token_index2.values())))

for i,sample in enumerate(samples):
    for j,character in list(enumerate(sample))[:max_length2]:
        index = token_index2.get(character)
        results2[i,j,index] = 1


print(results2[0,1])
#--------------------------------------------------------------------------------
#使用Keras实现one-hot
from keras.preprocessing.text import Tokenizer
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
#只考虑20个最常见的词。
tokenizer = Tokenizer(num_words=5)
#用samples构建单词索引
tokenizer.fit_on_texts(samples)
sequence = tokenizer.texts_to_sequences(samples)
print(sequence)
word_index = tokenizer.word_index
print(word_index)
#可以直接转化为ont-hot形式。不过是句子形式的
one_hot = tokenizer.texts_to_matrix(samples,mode='binary')
print(one_hot)



