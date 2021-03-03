import tensorflow as tf
from tensorflow.keras import layers
import math
import numpy as np

print(tf.__version__)
print(tf.keras.__version__)


model = tf.keras.Sequential()
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])
"""
分类是一个双曲线分成的三部分。蓝色部分对应的是x<=-(y^2+1)^(1/2)，其向量为[1,0,0]，
黄色部分对应的是x>=(y^2+1)^(1/2)，其向量为[0,0,1]，其余部分为x^2<y^2+1，对应向量为[0,1,0]

训练数据增加到40000个时，验证准确度已经接近1了
"""
train_x = np.array([[x, y] for x in range(-10, 10) for y in range(-10, 10)])
train_y = np.array(list(map(lambda v: [1, 0, 0] if v[0] <= -math.sqrt(v[1] * v[1] + 1) else (
    [0, 0, 1] if v[0] >= math.sqrt(v[1] * v[1] + 1) else [0, 1, 0]), train_x)))

print(train_x)
print(train_y)
"""从epochs=1到epoch=9，损失和准确度一直在提升，但是验证的值一直在下降，这是由于数据集太小了"""
history = model.fit(train_x, train_y, batch_size=1, epochs=5,
          validation_data=(np.array([[0, 0], [10, 0]]), np.array([[0, 1, 0], [0, 0, 1]])))

print(history.history)