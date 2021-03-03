import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
train_x = np.linspace(-1, 1, 100)
train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.3

# plt.plot(train_x, train_y, 'ro', label='origin data', c='b')
# plt.legend()
# plt.show()

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), name="bias")

Z = tf.multiply(X, W) + b
"""
规约计算的操作都会有降维的功能，在所有reduce_xxx系列操作函数中，都是以xxx的手段降维，
每个函数都有axis这个参数，即沿某个方向，使用xxx方法对输入的Tensor进行降维。
"""
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


def train():
    with tf.Session(config=config) as sess:
        with tf.device("/gpu:0"):
            sess.run(init)
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

            print("-------训练完成-------")
            print("cost:", sess.run(cost, feed_dict={X: train_x, Y: train_y}), "W:", sess.run(W), "b:", sess.run(b))

            # 参数现在是训练好的，输入X则有个对应的Y
            print("预测输入0.2的结果是：", sess.run(Z, feed_dict={X: 0.2}))

            pbsave(sess)

def pbsave(sess):
    graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,['weight'])
    with tf.gfile.FastGFile('../pbmodels/standard_process_classification2.pb', mode='wb') as f:
        f.write(graph.SerializeToString())
def save_method(sess):
    #因为要多次试验，所以每次保存前都清空目录
    ckpt_files = os.listdir("../ckptmodels/ckptmodels1")
    entire_path = [os.path.join("../ckptmodels/ckptmodels1/",file) for file in ckpt_files]
    for element in entire_path:
        os.remove(element)
    # 定义一个saver
    saver = tf.train.Saver()
    saver.save(sess, "../ckptmodels/standard_process_classification2.ckpt")

def save_moethod2(sess):
    ckpt_files = os.listdir("../ckptmodels")
    entire_path = [os.path.join("../ckptmodels/", file) for file in ckpt_files]
    for element in entire_path:
        os.remove(element)
    saver = tf.train.Saver({'weight1':W,'bias1':b})#代表将w变量的值放到weight名字中~~~
    """
    输出应该是:
    tensor_name:  bias1
    [0.00024379]
    tensor_name:  weight1
    [2.018459]

    """
    """
    类似的写法:
    saver = tf.train.Saver([W,b]}
    或者就使用W和b的name作为上面字典的key。利用字典推导式
    saver = tf.train.Saver({v.op.name:v for v in [W,b]})
    """
    saver.save(sess, "../ckptmodels/standard_process_classification2.ckpt")



# train()

def test1():
    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    def print_modelfile():
        print_tensors_in_checkpoint_file("../ckptmodels/standard_process_classification2.ckpt",None,True)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        graph = tf.get_default_graph()
        sess.run(init)
        saver.restore(sess,"../ckptmodels/ckptmodels1/standard_process_classification2.ckpt")
        print(sess.run("add:0",feed_dict={X:0.2}))
        print(Z)

def test2():
    from tensorflow.python.platform import gfile

    sess = tf.Session()
    with gfile.FastGFile('../pbmodels/standard_process_classification2.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图

    # 需要有一个初始化的过程
    sess.run(tf.global_variables_initializer())
    print(sess.run(sess.graph.get_operation_by_name("weight")))

test2()


