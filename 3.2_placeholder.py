import tensorflow as tf
import numpy as np


@tf.function
def add_two_values(x, y):
    return x + y


print(add_two_values(3, 4.5).numpy())
print(add_two_values(np.array([1, 3]), np.array([2, 4])).numpy())


@tf.function
def add_two_values_and_multiply_three(x, y):
    return 3 * add_two_values(x, y)


print(add_two_values_and_multiply_three(3, 4.5).numpy())