import tensorflow as tf

state = tf.Variable(0,name='counter')
c = tf.constant(1)
new_value = tf.add(state,c)
update = tf.assign(state,new_value)
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("../../../log/", sess.graph)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))