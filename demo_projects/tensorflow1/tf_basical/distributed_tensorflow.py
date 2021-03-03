"""
在大型的数据集上进行神经网络的训练，往往需要更大的运算资源，而且还要耗费若干天才能完成运算量。TensorFlow提供了一个可以分布式部署的模式，将一个训练任务拆成多个小任务，分配到不同的计算机上来完成协同运算，这样使用计算机群运算来代替单机计算，可以使训练时间大大缩短。
"""
#用在同一台机器上使用不同的端口来模拟分布式训练。
#在一台机器上开3个不同的端口，分别代表ps、chief、supervisors和worker。角色的名称用strjob_name表示
import tensorflow as tf
ps_hosts = "localhost:1681".split(",")#服务端
worker_hosts = "localhost:1682,localhost:1683".split(",")
job_name = "ps"
task_index = 0
cluster_rec = tf.train.ClusterSpec({'ps':ps_hosts,'worker':worker_hosts})

#创建server
server = tf.train.Server({'ps':ps_hosts,'worker':worker_hosts},job_name=job_name,task_index=task_index)

#ps角色使用join进行等待
if job_name == 'ps':
    print("wait......................")
    server.join()

#使用tf.device函数将全部的节点都放在当前任务下
#在tf.device函数中的任务是通过tf.train.replica_device_setter来指定的
#在tf.train.replica_device_setter中使用worker_device来定义具体任务名称；使用cluster的配置来指定角色及对应的IP地址，从而实现管理整个任务下的图节点
