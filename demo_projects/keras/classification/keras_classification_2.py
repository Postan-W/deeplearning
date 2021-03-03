from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from utility.graphic import training_and_validation_loss
import time
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("训练数据类型:",type(train_images))
print("训练数据标签:",type(train_labels))
print("训练标签:",train_labels)
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 25

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("onehot后的标签形状为：",train_labels.shape)

history = model.fit(train_images, train_labels, epochs=1, batch_size=128)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(model.name)
print(test_acc,test_loss)

#--------------------------------------------------
#保存为onnx
# from utility.to_onnx import converted_to_onnx
# converted_to_onnx(model,"keras","classification_2.1","../../../onnx_files/")
#保存模型文件
import os
current_file = str(os.path.basename(__file__)).split(".")[0]
model_path = "../models/"+str(time.strftime("%Y.%m.%d-%H.%M.%S"))+current_file+".h5"
model.save(model_path)