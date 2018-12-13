import numpy as np
import sklearn.preprocessing
import tensorflow as tf


class ConvolutionalNN(object):

    def __init__(self, n_pixels, n_classes=10, n_filters=3, kernel_size=3, name='cnn'):
        with tf.variable_scope('{name}/input'.format(name=name)):
            self.x = tf.placeholder(tf.float32, [None, n_pixels, n_pixels])
            self.X = tf.reshape(self.x, shape=[-1, n_pixels, n_pixels, 1])
            self.y = tf.placeholder(tf.int32, [None, n_classes])
            self.learning_rate = tf.placeholder(tf.float32, ())

        with tf.variable_scope('{name}/convolutions'.format(name=name)):
            self.conv = tf.layers.conv2d(
                self.X,
                filters=n_filters,
                kernel_size=kernel_size,
                padding='same',
                activation=tf.nn.relu,
            )

            self.pool = tf.layers.max_pooling2d(self.conv, pool_size=[2, 2], strides=2)

            pool_shape = tf.shape(self.pool)
            pool_flat = tf.reshape(self.pool, [-1, 14 * 14 * 3])

            dense = tf.layers.dense(inputs=pool_flat, units=200, activation=tf.nn.relu)
            self.logits = tf.layers.dense(inputs=dense, units=10)

        with tf.variable_scope('{name}/train'.format(name=name)):
            self.softmax = tf.losses.softmax_cross_entropy(self.y, self.logits)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.softmax)


def random_draw(x, y=None, batch_size=100):
    indices = np.random.randint(0, len(x), batch_size)
    if y is not None:
        return x[indices], y[indices]
    else:
        return x[indices]


def train_cnn(
    cnn,
    x_train,
    y_train,
    x_test,
    y_test,
    learning_rate,
    n_epochs,
    batch_size,
    print_every=1000,
    name='cnn'):
    saver = tf.train.Saver()

    encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)
    y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_oh = encoder.transform(y_test.reshape(-1, 1))

    best_loss = float('inf')
    save_path = f'./{name}.ckpt'
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        for epoch in range(n_epochs):
            x_batch, y_batch = random_draw(x_train, y_train_oh, batch_size)

            sess.run(
                cnn.train_op,
                feed_dict={
                    cnn.x: x_batch,
                    cnn.y: y_batch,
                    cnn.learning_rate: learning_rate,
                },
            )

            if epoch % print_every == 0:
                loss = sess.run(cnn.softmax, feed_dict={cnn.x: x_test, cnn.y: y_test_oh})
                print(f'{epoch} / {n_epochs}: loss = {loss}')

                if best_loss > loss:
                    save_path = saver.save(sess, f'./{name}.ckpt')
                    best_loss = loss

    return save_path
