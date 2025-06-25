import tensorflow as tf

a = tf.constant(1234)
b = tf.constant(5678)

@tf.function #텐서플로의 그래프 모드 최적화를 돕는 데코레이터
def add_op(a,b):
    return a+b

res = add_op(a,b).numpy()
print(res)