import tensorflow as tf

W = tf.Variable(tf.random.normal(shape=[1]))
b = tf.Variable(tf.random.normal(shape=[1]))


@tf.function
def linear_model(x):
    return W * x + b


@tf.function
def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred - y))  # MSE Loss Fuction


optimizer = tf.optimizers.SGD(0.01)  # 러닝레이트 = 0.01 로 지정함


@tf.function
def train_step(x, y):  # 파라미터를 업데이트 하는 연산 : train_step
    with tf.GradientTape() as tape:  # 미분
        y_pred = linear_model(x)
        loss = mse_loss(y_pred, y)
    gradients = tape.gradient(loss, [W, b])  # Loss Fuction을 각각 W와 b로 편미분 함
    optimizer.apply_gradients(zip(gradients, [W, b]))


# 학습을 위한 트레이닝 데이터
x_train = [1, 2, 3, 4]  # 입력 데이터
y_train = [2, 4, 6, 8]  # 출력 데이터

for i in range(1000):  # Gradient Descent를 1000번 수행함
    train_step(x_train, y_train)

x_test = [3.5, 5, 5.5, 6]  # 테스트를 위한 입력 데이터

# 예상되는 참값 : [7, 10, 11, 12]
print(linear_model(x_test).numpy())
