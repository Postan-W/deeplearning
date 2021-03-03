import tensorflow.contrib.eager as tfe
import tensorflow as tf
tfe.enable_eager_execution()

a = tf.constant([[1,2]])
b = tf.constant([[3],[4]])
c = tf.matmul(a,b)
print(c)