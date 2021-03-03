import tensorflow as tf

c1 = tf.constant(3)
c2 = tf.constant(4)

with tf.Session() as sess:
    print(sess.run(c1*c2))

p1 = tf.placeholder(tf.int16)
p2 = tf.placeholder(tf.int16)

add = tf.add(p1,p2)
mul = tf.multiply(p1,p2)

with tf.Session() as sess:
    print(sess.run(add,feed_dict={p1:3,p2:4}))
    print(sess.run(mul,feed_dict={p1:4,p2:5}))
    print(sess.run([mul,add],feed_dict={p1:6,p2:7}))
