import tensorflow as tf
import numpy as np
alpha = 0.01 #学习率
epoch = 5 #轮数

x_data = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
y_data = np.array([[1],[2],[3],[4],[5]])
x = tf.placeholder(tf.float32,(5,3))
y = tf.placeholder(tf.float32,(5,1))

w = tf.get_variable("weights",(3,1),initializer=tf.constant_initializer())

y_predict = tf.matmul(x,w)


loss_op = 1 / 2*len(x_data) * tf.matmul((y_predict-y),(y_predict-y),transpose_a=True)

opt = tf.train.GradientDescentOptimizer(learning_rate=alpha)

train_op = opt.minimize(loss_op)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epoch):
        sess.run(train_op,feed_dict={x:x_data,y:y_data})
        loss,w = sess.run([loss_op,w],feed_dict={x:x_data,y:y_data})
        #log_str = "Epoch:%d \t loss:%.4f w:"%(e,loss)
        print(loss,w)


