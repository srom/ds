import numpy as np
import tensorflow as tf

from . import EncoderDecoder


class EncoderDecoderTest(tf.test.TestCase):

    def test_lstm_network(self):
        network = EncoderDecoder(
            n_inputs=4,
            n_outputs=4,
            n_steps=2,
            n_neurons=10,
            learning_rate=0.1,
            batch_size=2,
        )

        X = np.array([[
            [4.3, 2.3, 12.9, 99.88],
            [4.4, 2.2, 8.0, 99.94],
        ]])
        Y = np.array([[
            [4.4, 2.2, 8.0, 99.94],
            [3.8, 1.6, 6.4, 100.0],
        ]])

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            network.training_op.run(feed_dict={network.X: X, network.Y: Y})
            mse = network.loss.eval(feed_dict={network.X: X, network.Y: Y})
            self.assertIsNotNone(mse)


if __name__ == '__main__':
    tf.test.main()
