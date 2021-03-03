import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
train_x = np.linspace(-1, 1, 100)
train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.3

plt.plot(train_x, train_y, 'ro', label='origin data', c='b')
plt.legend()
plt.show()

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), name="bias")

Z = tf.multiply(X, W) + b
tf.summary.histogram('Z',Z)#将预测值以直方图形式展示
cost = tf.reduce_mean(tf.square(Y - Z))
tf.summary.scalar('loss',cost)
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


def train():
    with tf.Session(config=config) as sess:
        with tf.device("/gpu:0"):
            sess.run(init)
            #合并所有summary
            merged = tf.summary.merge_all()
            #创建writer用于写文件
            writer = tf.summary.FileWriter("./log/",sess.graph)#如果只操作到这，那么tensorboard上只有一副流图，没有其他的
            plotdata = {"batchsize": [], "loss": []}
            for epoch in range(epochs):
                for (x, y) in zip(train_x, train_y):  # 这就相当于batchsize=1
                    sess.run(optimizer, feed_dict={X: x, Y: y})

                if epoch % display_step == 0:
                    loss = sess.run(cost, feed_dict={X: train_x, Y: train_y})
                    print("Epoch:", epoch + 1, "cost:", loss, "W:", sess.run(W), "b:", sess.run(b))
                    if not (loss == "NA"):
                        plotdata["batchsize"].append(epoch)
                        plotdata["loss"].append(loss)
                summary_str = sess.run(merged,feed_dict={X:x,Y:y})
                writer.add_summary(summary_str,epoch)
            print("-------训练完成-------")
            print("cost:", sess.run(cost, feed_dict={X: train_x, Y: train_y}), "W:", sess.run(W), "b:", sess.run(b))

            # 参数现在是训练好的，输入X则有个对应的Y
            print("预测输入0.2的结果是：", sess.run(Z, feed_dict={X: 0.2}))


train()