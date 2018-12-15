import tensorflow as tf
import numpy as np

layer_dims = [100, 200, 100, 50, 25, 3]


def get_weights(input_dim, output_dim):
    return tf.Variable(
        tf.truncated_normal([input_dim, output_dim], stddev=0.5),
        trainable=True, dtype=tf.float32, validate_shape=True,
        expected_shape=[input_dim, output_dim])


class NaiveNet(object):
    input_dim = 100  # 需要根据具体情况更改
    label_dim = 3
    regularization_rate = 0.0001
    learning_rate = 0.001
    decay_rate = 0.999

    def __init__(self):
        self.input_x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="input_x")
        self.label_y = tf.placeholder(tf.float32, shape=[None, self.label_dim], name="label_y")
        weights = list()
        weights_length = len(layer_dims) - 1
        i = 0
        while i < weights_length:
            weights.append(get_weights(layer_dims[i], layer_dims[i+1]))
            i = i + 1
        biases = list()
        i = 1
        while i < len(layer_dims):
            biases.append(tf.Variable(tf.constant(0.0, shape=[layer_dims[i]])))
            i = i + 1
        # Forward pass
        assert len(biases) == len(weights), "length of biases should be equal to weights"
        out_value = self.input_x
        for i in range(len(weights) - 1):
            out_value = tf.nn.relu(out_value @ weights[i] + biases[i])
        self.predict_y = out_value @ weights[-1] + biases[-1]
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label_y, logits=self.predict_y)
        self.loss = tf.reduce_mean(self.loss)
        # Regularization.
        regularizer = tf.contrib.layers.l2_regularizer(self.regularization_rate)
        regularization = regularizer(weights[-1])
        for i in range(len(weights) - 1):
            regularization = regularization + regularizer(weights[i])
        self.loss += regularization
        self.global_step = tf.Variable(0, trainable=False)
        # For validations:
        correct_predictions = tf.equal(tf.argmax(self.predict_y, 1), tf.argmax(self.label_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        # Create Session
        self.sess = tf.Session()

    def train(self, step_num, data_set, validation_set):
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   self.global_step,
                                                   data_set.num_examples,
                                                   self.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
        for i in range(step_num):
            if i % 500 == 0:
                validate_result = self.sess.run(self.accuracy,
                                                feed_dict={self.input_x: validation_set.xs,
                                                           self.label_y: validation_set.labels})
                print("After %d training step(s), validation accuracy is %g "
                      % (i, validate_result))
            self.sess.run(optimizer, feed_dict={self.input_x: data_set.xs,
                                                self.label_y: data_set.labels})


class DataSet(object):
    def __init__(self, num):
        self.num_examples = num
        self.xs = np.floor(np.random.random(size=(num, 100)) + 0.5)
        self.labels = np.floor(np.random.random(size=(num, 3)) + 0.5)
        print(self.xs)

    def next_batch(self, batch_num):
        xs = np.array([batch_num, 2826])
        labels = np.array([batch_num, 3])
        return xs, labels


if __name__ == "__main__":
    net = NaiveNet()
    fake = DataSet(1000)
    net.train(40000, fake, fake)
