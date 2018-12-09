import tensorflow as tf


class BinaryClassifier(object):

    def __init__(self, input_size, hidden_layer_size):
        self.x = tf.placeholder(tf.float32, [None, input_size])
        self.y = tf.placeholder(tf.float32, [None])
        self.learning_rate = tf.placeholder(tf.float32, ())

        hidden = tf.layers.dense(self.x, hidden_layer_size, tf.nn.relu)
        self.outcome = tf.squeeze(tf.layers.dense(hidden, 1, tf.nn.sigmoid))

        self.loss = tf.losses.log_loss(self.y, self.outcome)

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
