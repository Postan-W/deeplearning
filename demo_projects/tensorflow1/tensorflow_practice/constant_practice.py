import tensorflow as tf
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
product = tf.matmul(matrix1,matrix2,name='product')

sess = tf.Session()


writer = tf.summary.FileWriter("../../../log/", sess.graph)

sess.run(product)