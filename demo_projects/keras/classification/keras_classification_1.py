#这里处理一个单标签多分类问题，即有2个以上类别，但单个数据只属于其中一个
from keras.datasets import reuters
from keras import models
from keras import layers
from utility.vectorization import vectorization
from utility.graphic import training_and_validation_loss
from utility.timer import start_time,end_time
from keras.utils.np_utils import to_categorical
import numpy as np
from utility import all_callbacks
import time
start_time = start_time()
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print(train_labels[np.argmin(train_labels)])
word_index = reuters.get_word_index()
"""part of word_index output:
      {'mdbl': 10996, 'fawc': 16260, 'degussa': 12089, 'woods': 8803, 'hanging': 13796, 'localized': 20672}  
"""
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
"""
decoded_newswire:
    ? ? ? said as a result of its december acquisition of space co it expects earnings per share in 1987 of 1 15 to 1 30 dlrs per share up from 70 cts in 1986 the company said pretax net should rise to nine to 10 mln dlrs from six mln dlrs in 1986 and rental operation revenues to 19 to 22 mln dlrs from 12 5 mln dlrs it said cash flow per share this year should be 2 50 to three dlrs reuter 3
"""
#将数据向量化(这里的dimension肯定是和第一层的input_shape是一致的)
x_train = vectorization(train_data,10000)
x_test = vectorization(test_data,10000)
#将标签向量化,因为有46个类别，所以dim=46
#one_hot_train_labels = vectorization(train_labels,dimension=46)
#one_hot_test_labels = vectorization(test_labels,dimension=46)

#上面的标签向量化可以用Keras自带的函数实现
one_hot_train_labels = to_categorical(train_labels) 
one_hot_test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]
history = model.fit(partial_x_train, partial_y_train, epochs=30, batch_size=512, validation_data=(x_val,y_val))
training_and_validation_loss(history.history)
end_time = end_time()
delay = end_time - start_time
print("测试集上的loss和accuracy:",model.evaluate(x_test,one_hot_test_labels))
print("总耗时：",delay,"秒")
#下面来预测
predictions = model.predict(x_test)
#一共有46类，每一个样本的预测向量维度一定是46。总和一定为1。所属类别属于概率最大的那个
print(predictions[0].shape, np.sum(predictions[0]),np.argmax(predictions[0]))

#保存模型文件
import os
current_file = str(os.path.basename(__file__)).split(".")[0]
model_path = "../models/"+str(time.strftime("%Y.%m.%d-%H.%M.%S"))+current_file+".h5"

#--------------------------------------------------------
#保存为onnx
import keras2onnx
from utility.to_onnx import converted_to_onnx
converted_to_onnx(model,"keras","classification_1","../../../onnx_files/")
#保存为h5格式
model.save(model_path)
