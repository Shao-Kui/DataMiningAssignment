import tensorflow as tf
from data_set import DataSet
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np
diag_dim = 2253
label_dim = 3
layer_dims_branch1 = [573, 44]
layer_dims_branch2 = [diag_dim, 3]
layer_dims_merge = [layer_dims_branch1[-1] + layer_dims_branch2[-1], 50, label_dim]
batch_num = 1000
learning_rate = 0.001

accuracies = []
precisions = []
recalls = []
F1s = []


def get_weights(input_dim, output_dim):
    return tf.Variable(
        tf.truncated_normal([input_dim, output_dim], mean=0.0, stddev=0.1),
        trainable=True, dtype=tf.float32, validate_shape=True,
        expected_shape=[input_dim, output_dim])


class NaiveNet(object):
    input_dim = 573  # 需要根据具体情况更改
    label_dim = 3
    regularization_rate = 0.00001
    decay_rate = 0.99

    def __init__(self):
        self.best_accu = 0
        self.best_precision = 0
        self.best_recall = 0
        self.best_f1 = 0
        self.max_test_accuracy = 0.0
        self.input_x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="input_x")
        self.diag_x = tf.placeholder(tf.float32, shape=[None, diag_dim], name="diag_x")
        self.label_y = tf.placeholder(tf.float32, shape=[None, self.label_dim], name="label_y")
        self.keep_prob = tf.placeholder(tf.float32)
        # Branch1 weights
        weights = list()
        weights_length = len(layer_dims_branch1) - 1
        i = 0
        while i < weights_length:
            weights.append(get_weights(layer_dims_branch1[i], layer_dims_branch1[i+1]))
            i = i + 1
        biases = list()
        i = 1
        while i < len(layer_dims_branch1):
            biases.append(tf.Variable(tf.constant(0.0, shape=[layer_dims_branch1[i]])))
            i = i + 1
        # Branch2 weights
        weights_branch2 = list()
        weights_length = len(layer_dims_branch2) - 1
        i = 0
        while i < weights_length:
            weights_branch2.append(get_weights(layer_dims_branch2[i], layer_dims_branch2[i+1]))
            i = i + 1
        biases_branch2 = list()
        i = 1
        while i < len(layer_dims_branch2):
            biases_branch2.append(tf.Variable(tf.constant(0.0, shape=[layer_dims_branch2[i]])))
            i = i + 1
        # Merge weights
        weights_merge = list()
        weights_length = len(layer_dims_merge) - 1
        i = 0
        while i < weights_length:
            weights_merge.append(get_weights(layer_dims_merge[i], layer_dims_merge[i+1]))
            i = i + 1
        biases_merge = list()
        i = 1
        while i < len(layer_dims_merge):
            biases_merge.append(tf.Variable(tf.constant(0.0, shape=[layer_dims_merge[i]])))
            i = i + 1
        # Forward pass
        assert len(biases) == len(weights), "length of biases should be equal to weights"
        out_value = self.input_x
        for i in range(len(weights)):
            out_value = tf.nn.leaky_relu(out_value @ weights[i] + biases[i])
        out_value_branch2 = self.diag_x
        for i in range(len(weights_branch2)):
            out_value_branch2 = tf.nn.leaky_relu(out_value_branch2 @ weights_branch2[i] + biases_branch2[i])
        out_value = tf.concat([out_value, out_value_branch2], 1)
        for i in range(len(weights_merge) - 1):
            out_value = tf.nn.leaky_relu(out_value @ weights_merge[i] + biases_merge[i])
        self.predict_y = out_value @ weights_merge[-1] + biases_merge[-1]
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label_y, logits=self.predict_y)
        self.loss = tf.reduce_mean(self.loss)\
        # Regularization.
        regularizer = tf.contrib.layers.l2_regularizer(self.regularization_rate)
        regularization = regularizer(weights[-1])
        for i in range(len(weights) - 1):
            regularization = regularization + regularizer(weights[i])
        self.loss += regularization
        self.global_step = tf.Variable(0, trainable=False)
        # For validations:
        self.predict_y_label = tf.argmax(self.predict_y, 1)
        correct_predictions = tf.equal(tf.argmax(self.predict_y, 1), tf.argmax(self.label_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        # Create Session
        self.sess = tf.Session()

    def train(self, step_num, data_set, validation_set):
        learning_rate_decay = tf.train.exponential_decay(learning_rate,
                                                   self.global_step,
                                                   data_set.num_examples,
                                                   self.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate_decay).minimize(self.loss, global_step=self.global_step)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss, global_step=self.global_step)
        # if os.path.isfile("./tmp/model.ckpt.meta"):
        #     saver = tf.train.Saver()
        #     saver.restore(self.sess, "./tmp/model.ckpt")
        #     print("Model restored.")
        # else:
        #     self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        for i in range(step_num):
            if i % 1 == 0:
                # validate_result = self.sess.run(self.accuracy,
                #                                 feed_dict={self.input_x: validation_set.xs,
                #                                            self.diag_x: validation_set.diags,
                #                                            self.label_y: validation_set.labels,
                #                                            self.keep_prob: 0.4})
                # train_result = self.sess.run(self.accuracy,
                #                              feed_dict={self.input_x: data_set.xs,
                #                                         self.diag_x: data_set.diags,
                #                                         self.label_y: data_set.labels,
                #                                         self.keep_prob: 0.4})
                # print("After %d training step(s), validation accuracy is %g "
                #       % (i, validate_result))
                # print("After %d training step(s), training   accuracy is %g "
                #       % (i, train_result))
                predict_y = self.predict(validation_set)
                test_y = np.argmax(validation_set.labels, axis=1)
                precision = precision_score(test_y, predict_y, average='macro')
                accu = accuracy_score(test_y, predict_y)
                recall = recall_score(test_y, predict_y, average='macro')
                F1 = (2.0 * recall * precision) / (recall + precision)
                if F1 > self.best_f1:
                    self.best_f1 = F1
                    self.best_accu = accu
                    self.best_precision = precision
                    self.best_recall = recall
                    print("After %d training step(s): " % i)
                    print("Accuracy: ", accu)
                    print("Precision: ", precision)
                    print("Recall: ", recall)
                    print("F1: ", F1)
                    print("---------------------------------------------")
                # if self.max_test_accuracy < validate_result:
                #     saver = tf.train.Saver()
                #     self.max_test_accuracy = validate_result
                #     save_path = saver.save(self.sess, "./tmp/model.ckpt")
                #     print("Model saved in path: %s" % save_path)
                #     print("Current max accuracy is ", validate_result)
            xs, diags, labels = data_set.next_batch(batch_num)
            self.sess.run(optimizer, feed_dict={self.input_x: xs,
                                                self.label_y: labels,
                                                self.diag_x: diags,
                                                self.keep_prob: 0.4})

    def predict(self, dataset):
        return self.sess.run(self.predict_y_label,
                                        feed_dict={self.input_x: dataset.xs,
                                                   self.diag_x: dataset.diags,
                                                   self.label_y: dataset.labels,
                                                   self.keep_prob: 0.4})


if __name__ == "__main__":
    for i in range(10):
        print("start with dataset: ", i)
        net = NaiveNet()
        test_set = DataSet(i, prefix="test")
        net.train(1000, DataSet(i), test_set)
        accuracies.append(net.best_accu)
        precisions.append(net.best_precision)
        recalls.append(net.best_recall)
        F1s.append(net.best_f1)
    print("Final Average Accuracy: ", np.average(accuracies))
    print("Final Average Precisions: ", np.average(precisions))
    print("Final Average Recalls: ", np.average(recalls))
    print("Final Average F1s: ", np.average(F1s))
