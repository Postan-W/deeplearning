
import tensorflow as tf
g = tf.get_default_graph()
a = tf.constant([[1.0,2.0]])
b = tf.constant([[1.0],[3.0]])
print(a)
print(b)
tensor1 = tf.matmul(a,b,name='exampleop')
print(tensor1.name)
print(tensor1)
test = g.get_tensor_by_name("exampleop:0")
print(test)

print(tensor1.op.name)

print(g.get_operation_by_name("exampleop"))

with tf.Session() as sess:
    test = sess.run(test)
    print(test)
    test = tf.get_default_graph().get_tensor_by_name("exampleop:0")
    print(test)

print(g.get_operations())#获取所有元素，包括张量和操作

print(g.as_graph_element(a,allow_tensor=True,allow_operation=True))#获取对象

