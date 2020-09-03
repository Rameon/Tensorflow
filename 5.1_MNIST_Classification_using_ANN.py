import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])
x_train, x_test = x_train / 255., x_test / 255.

y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)

learning_rate = 0.001
num_epochs = 30         # 학습횟수
batch_size = 256        # 배치개수
display_step = 1        # Loss Function 출력 주기
input_size = 784        # 28 * 28
hidden1_size = 256
hidden2_size = 256
output_size = 10

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(60000).batch(batch_size)


class ANN(object):
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


@tf.function
def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))


optimizer = tf.optimizers.Adam(learning_rate)


@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = cross_entropy_loss(y_pred, y)
    gradients = tape.gradient(loss, vars(model).values())
    optimizer.apply_gradients(zip(gradients, vars(model).values()))


@tf.function
def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


ANN_model = ANN()

for epoch in range(num_epochs):
    average_loss = 0.
    total_batch = int(x_train.shape[0] / batch_size)
    for batch_x, batch_y in train_data:
        _, current_loss = train_step(ANN_model, batch_x, batch_y), cross_entropy_loss(ANN_model(batch_x), batch_y)
        average_loss += current_loss / total_batch
    if epoch % display_step == 0:
        print("반복(Epoch) : %d, Loss Function(Loss) : %f" % ((epoch+1), average_loss))

print("정확도(Accuracy) : %f" % compute_accuracy(ANN_model(x_test), y_test))       # 정확도 : 약 94%
