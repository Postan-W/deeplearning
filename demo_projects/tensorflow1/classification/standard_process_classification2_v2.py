import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
train_x = np.linspace(-1, 1, 100)
train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.3



X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), name="bias")

Z = tf.multiply(X, W) + b

cost = tf.reduce_mean(tf.square(Y - Z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# 初始化变量
init = tf.global_variables_initializer()

epochs = 10
display_step = 2
"""
● log_device_placement=True：是否打印设备分配日志。
● allow_soft_placement=True：如果指定的设备不存在，允许TF自动分配设备。

"""
gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.7)
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
# 使用allow_growth option，刚开始会分配少量的GPU容量，然后按需慢慢地增加，由于不会释放内存，所以会导致碎片。
# 也可以用config.allow_growth_option = True代替GPUOptions(allow_growth=True)

saver = tf.train.Saver(max_to_keep=2)
def train():
    with tf.Session(config=config) as sess:
        with tf.device("/gpu:0"):
            sess.run(init)
            plotdata = {"batchsize": [], "loss": []}
            for epoch in range(epochs):
                for (x, y) in zip(train_x, train_y):  # 这就相当于batchsize=1
                    sess.run(optimizer, feed_dict={X: x, Y: y})

                if (epoch+1) % display_step == 0:
                    loss = sess.run(cost, feed_dict={X: train_x, Y: train_y})
                    print("Epoch:", epoch + 1, "cost:", loss, "W:", sess.run(W), "b:", sess.run(b))
                    if not (loss == "NA"):
                        plotdata["batchsize"].append(epoch)
                        plotdata["loss"].append(loss)
                    #每2个轮次保存一次，max_to_keep限定了最多保存两次
                    saver.save(sess,"../ckptmodels/ckptmodels2/epochcheckpoint.ckpt",global_step=epoch+1)
            print("-------训练完成-------")
            print("cost:", sess.run(cost, feed_dict={X: train_x, Y: train_y}), "W:", sess.run(W), "b:", sess.run(b))

            # 参数现在是训练好的，输入X则有个对应的Y
            print("预测输入0.2的结果是：", sess.run(Z, feed_dict={X: 0.2}))

# train()

#使用的时候请注意也要加上对应的轮次后缀。这里使用第8轮的
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess,"../ckptmodels/ckptmodels2/epochcheckpoint.ckpt-8")
    print("第8轮输入0.2看看输出是：",sess.run(Z,feed_dict={X:0.2}))
