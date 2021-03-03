#下载tf.keras.applications包中已有的模型
#下载的默认位置时~/.keras/models/
import tensorflow as tf
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192,192,3),include_top=False)
#参数固定
mobile_net.trainable = False
