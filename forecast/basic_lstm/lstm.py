import tensorflow as tf


class LSTMNetwork(object):

    def __init__(
            self,
            n_inputs,
            n_outputs,
            n_steps,
            n_neurons,
            learning_rate,
            adam_epsilon=1e8,
            name='lstm',
    ):
        with tf.variable_scope(f'{name}/input'):
            self.X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='X')
            self.Y = tf.placeholder(tf.float32, [None, n_steps, n_outputs], name='Y')

        with tf.variable_scope(f'{name}/rnn'):
            cell = tf.contrib.rnn.OutputProjectionWrapper(
                tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons),
                output_size=n_outputs
            )
            self.rnn_outputs, _ = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32, scope='dynamic_rnn')

        with tf.variable_scope(f'{name}/loss'):
            self.diff = tf.subtract(self.rnn_outputs, self.Y, name='diff')
            self.loss = tf.reduce_mean(tf.square(self.diff), name='mse')

        with tf.variable_scope(f'{name}/train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=adam_epsilon)
            self.training_op = optimizer.minimize(self.loss)

        with tf.variable_scope('summary'):
            tf.summary.scalar('mse', tf.reduce_mean(self.loss))
            tf.summary.scalar('diff', tf.reduce_mean(self.diff))

        self.summary = tf.summary.merge_all()
