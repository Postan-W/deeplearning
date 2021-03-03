#卷积神经网络
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from utility import all_callbacks
from keras.utils import plot_model
import time
#python3中plot_model需要pydot-ng模块的支持，请事先install

(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
print(train_images.shape)
print(train_images[1].shape)
train_images = train_images.reshape((60000, 28, 28, 1))
print(train_images.shape)
print(train_images[1].shape)
train_images = train_images.astype('float32') / 255
print(train_images[1].shape)
print(train_images[1])

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=1, batch_size=64,)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)

print(model.predict(train_images[:1]))
#保存模型文件
import os
current_file = str(os.path.basename(__file__)).split(".")[0]
model_path = "../models/"+str(time.strftime("%Y.%m.%d-%H.%M.%S"))+current_file+".h5"
model.save(model_path)
