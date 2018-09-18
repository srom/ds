import tensorflow as tf


class LSTMNetwork(object):

    def __init__(self, n_inputs, n_outputs, n_steps, n_neurons, learning_rate, adam_epsilon=1e8):
        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
            self.Y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

        with tf.name_scope('rnn'):
            cell = tf.contrib.rnn.OutputProjectionWrapper(
                tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons),
                output_size=n_outputs
            )
            rnn_outputs, states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(rnn_outputs - self.Y))

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=adam_epsilon)
            self.training_op = optimizer.minimize(self.loss)
