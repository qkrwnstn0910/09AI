import tensorflow as tf

a = tf.constant([1,2,3], dtype=tf.int32)
b = tf.constant(2, dtype=tf.int32)
x2_op = a*b
print(x2_op.numpy())

a=tf.constant([10,20,10], dtype=tf.int32)
c2_op = a*b
print(x2_op.numpy())