import tensorflow as tf

w = tf.Variable(initial_value=tf.random_normal(shape=(1,4),mean=100,stddev=0.35),name='w')
b = tf.Variable(tf.zeros([4]),name='b')
print([w,b])
sess = tf.Session()

sess.run(tf.global_variables_initializer())

print(sess.run([w,b]))