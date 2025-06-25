import tensorflow as tf
import datetime as da

log_dir = "log_dir/"+ da.datetime.now().strftime("%Y%m%d-%H$M%S")
writer = tf.summary.create_file_writer(log_dir)

@tf.function
def compute():
    a = tf.constant(100, name="a")
    b = tf.constant(200, name="b")

    v= a*b
    return v

result = compute()
print("계산 결과:", result.numpy())

with writer.as_default():
    tf.summary.graph(compute.get_concrete_function().graph)

print(f"TensorBoard write ok : {log_dir}    ")