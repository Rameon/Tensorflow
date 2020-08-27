import tensorflow as tf

# linear regression model Wx + b
W = tf.Variable(tf.random.normal(shape=[1]))
b = tf.Variable(tf.random.normal(shape=[1]))


# Linear Regression Model
@tf.function
def linear_model(x):
  return W*x + b


# MSE Loss Function
@tf.function
def mse_loss(y_pred, y):
  return tf.reduce_mean(tf.square(y_pred - y))


# Gradient Descent Optimizer -> 0.01
optimizer = tf.optimizers.SGD(0.01)


# 텐서보드 summary 정보들을 저장할 폴더 경로를 설정합니다.
summary_writer = tf.summary.create_file_writer('./tensorboard_log')


# Optimization Function
@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
    y_pred = linear_model(x)
    loss = mse_loss(y_pred, y)
  with summary_writer.as_default():
    tf.summary.scalar('loss', loss, step=optimizer.iterations)
  gradients = tape.gradient(loss, [W, b])
  optimizer.apply_gradients(zip(gradients, [W, b]))


# Training을 위한 input data, output data
x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]


# Gradient Descent 1000번 수행함
for i in range(1000):
  train_step(x_train, y_train)


# Test를 위한 input data
x_test = [3.5, 5, 5.5, 6]

# 테스트 데이터를 이용해 학습된 선형회귀 모델이 데이터의 경향성(y=2x)을 잘 학습했는지 측정합니다.
# 예상되는 참값 : [7, 10, 11, 12]
print(linear_model(x_test).numpy())