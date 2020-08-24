import tensorflow as tf

node1 = tf.constant(3.0, dtype = tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

print(node1.numpy(), node2.numpy())

node3 = tf.add(node1, node2)
print("node3 : ", node3)
print("node3.numpy() : ", node3.numpy())