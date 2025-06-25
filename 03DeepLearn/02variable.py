import tensorflow as tf

a= tf.constant(120,name="a")
b= tf.constant(130,name="b")
c= tf.constant(140,name="c")

#tf.variable  정의. 값이 변경 가능하 텐서 정의
v = tf.Variable(0, name='v')

calc_op = a+b+c
v.assign(calc_op)

print(v.numpy())