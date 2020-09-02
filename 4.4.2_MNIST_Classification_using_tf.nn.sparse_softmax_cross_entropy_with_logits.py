# 1. Tensorflow 라이브러리를 임포트함
import tensorflow as tf

# 2. MNIST data를 다운로드함
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 3. data type 변경 -> 차원 변경(flattening) -> Normalizing -> One-hot Encoding
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')  		# 이미지들을 float32 data type으로 변경함
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])  	# 28*28 형태의 이미지를 784 차원으로 flattening함
x_train, x_test = x_train / 255., x_test / 255.  							# [0, 255] 사이의 값을 [0, 1] 사이의 값으로 normalize함

y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10) 	# 레이블 데이터에 one-hot Encoding을 적용함

# 4. tf.data API를 이용하여 데이터를 섞고 Batch 형태로 가져옴
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(60000).batch(100)
train_data_iter = iter(train_data)

# 5. Softmax Regression model을 위한 tf.Variable 들을 정의함
W = tf.Variable(tf.zeros(shape=[784, 10]))
b = tf.Variable(tf.zeros(shape=[10]))


# 6. Softmax Regression model을 정의함
@tf.function
def softmax_regression(x):
    logits = tf.matmul(x, W) + b
    return tf.nn.softmax(logits)


# 7. cross-entropy Loss Function을 정의함
@tf.function
def cross_entropy_loss(y_pred, y):
    # tf.nn.softmax_cross_entropy_with_logits API를 사용함 -> error
    # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    # Cross Entropy Loss Function을 직접 사용함
    return tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(y_pred), axis=[1]))


# 8. model의 정확도를 출력하는 함수를 정의함
@tf.function
def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


# 9. Optimization을 위한 Gradient Descent Optimizer를 정의함
optimizer = tf.optimizers.SGD(0.5)


# 10. Optimization을 위한 함수를 정의함
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = softmax_regression(x)
        loss = cross_entropy_loss(y_pred, y)
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))


# 11. 1000번 반복을 수행하면서 Optimization을 수행함
for i in range(1000):
    batch_xs, batch_ys = next(train_data_iter)
    train_step(batch_xs, batch_ys)


# 12. Train 후 학습된 model의 정확도를 출력함
print("정확도(Accuracy) : %f" % compute_accuracy(softmax_regression(x_test), y_test))  # 정확도 : 약 91%
