import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()      # MNIST data를 다운로드함
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')           # 이미지들을 float32 data type으로 변경함
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])         # 28*28 형태의 이미지를 784차원으로 flattening함
x_train, x_test = x_train / 255., x_test / 255.                                 # [0, 255] 사이의 값을 [0,1] 사이의 값으로 Normalize함 

y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)   # 레이블 데이터에 One-hot Encoding을 적용함

learning_rate = 0.001
num_epochs = 30         # 학습횟수
batch_size = 256        # 배치개수
display_step = 1        # Loss Function 출력 주기
input_size = 784        # 28 * 28
hidden1_size = 256
hidden2_size = 256
output_size = 10

# tf.data API를 이용하여 데이터를 섞고 batch 형태로 가져옴
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(60000).batch(batch_size)


# ANN model을 정의함
class ANN(object):
    # ANN model을 위한 tf.Variable들을 정의함
    def __init__(self):
        self.W1 = tf.Variable(tf.random.normal(shape=[input_size, hidden1_size]))
        self.b1 = tf.Variable(tf.random.normal(shape=[hidden1_size]))
        self.W2 = tf.Variable(tf.random.normal(shape=[hidden1_size, hidden2_size]))
        self.b2 = tf.Variable(tf.random.normal(shape=[hidden2_size]))
        self.W_output = tf.Variable(tf.random.normal(shape=[hidden2_size, output_size]))
        self.b_output = tf.Variable(tf.random.normal(shape=[output_size]))

    def __call__(self, x):
        H1_output = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        H2_output = tf.nn.relu(tf.matmul(H1_output, self.W2) + self.b2)
        logits = tf.matmul(H2_output, self.W_output) + self.b_output

        return logits


# cross-Entropy Loss Function을 정의함
@tf.function
def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# 최적화를 위한 Adam optimizer를 정의함
optimizer = tf.optimizers.Adam(learning_rate)


# 최적화를 위한 Function을 정의함
@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = cross_entropy_loss(y_pred, y)
    gradients = tape.gradient(loss, vars(model).values())
    optimizer.apply_gradients(zip(gradients, vars(model).values()))


# model의 정확도를 출력하는 함수를 정의함
@tf.function
def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


# ANN model을 선언함
ANN_model = ANN()

# 지정된 횟수만큼 최적화를 수행함
for epoch in range(num_epochs):
    average_loss = 0.
    total_batch = int(x_train.shape[0] / batch_size)
    for batch_x, batch_y in train_data:         # 모든 batch들에 대해서 최적화를 수행함
        _, current_loss = train_step(ANN_model, batch_x, batch_y), cross_entropy_loss(ANN_model(batch_x), batch_y)      # optimizer를 실행해서 파라미터를 업데이트함
        average_loss += current_loss / total_batch                                          # 평균 손실을 측정함
    if epoch % display_step == 0:               # 지정된 epoch마다 학습결과를 출력함
        print("반복(Epoch) : %d, Loss Function(Loss) : %f" % ((epoch+1), average_loss))

# test data를 이용해서 학습된 model이 얼마나 정확한지 정확도(accuracy)를 출력함
print("정확도(Accuracy) : %f" % compute_accuracy(ANN_model(x_test), y_test))       # 정확도 : 약 94%

