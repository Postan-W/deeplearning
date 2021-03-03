#根据矩阵乘法，从输入层到嵌入层相当于简单的查找，更形象的说相当于字典查找
from keras.datasets import imdb
from keras import preprocessing
from keras import models
from keras.layers import Flatten,Embedding,Dense
#句子中只包含全部语料最常出现的10000的词，在此之外的忽略掉
max_features = 10000
#句子长度限定为20，即保留句子中20个最常出现的词，不足部分用零补充
maxlen = 20
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(x_train[0])
#embedding层的激活形状为(samples,maxlen,8)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
print(x_train[0])
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
model = models.Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,epochs=10,batch_size=32,validation_split=0.2)