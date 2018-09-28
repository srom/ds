import unittest

import numpy as np

from .weather import next_batch


class TestNextBatch(unittest.TestCase):

    def test_next_batch(self):
        np.random.seed(444)

        X = np.array([
            [-2.0, 4.0, 5.0, 1.0],
            [-2.4, 4.1, 5.2, 1.7],
            [2.0, 5.0, 4.0, 1.9],
            [2.0, 4.0, 5.0, 1.9],
            [-3.0, 4.0, 5.0, 1.0],
            [-5.4, 4.1, 5.2, 1.7],
            [5.0, 5.0, 4.0, 1.9],
            [6.0, 4.0, 5.0, 1.9],
        ])

        batch_X, batch_Y = next_batch(X, batch_size=2, time_steps=2)

        self.assertEqual(2, len(batch_X))
        self.assertEqual(2, len(batch_X[0]))

        self.assertEqual(2, len(batch_Y))
        self.assertEqual(2, len(batch_Y[0]))

        self.assertTrue(np.array_equal(
            np.array([
                [2.0, 4.0, 5.0, 1.9],
                [-3.0, 4.0, 5.0, 1.0],
            ]),
            batch_X[0]
        ))
        self.assertTrue(np.array_equal(
            np.array([
                [-3.0, 4.0, 5.0, 1.0],
                [-5.4, 4.1, 5.2, 1.7],
            ]),
            batch_Y[0]
        ))

        self.assertTrue(np.array_equal(
            np.array([
                [-2.0, 4.0, 5.0, 1.0],
                [-2.4, 4.1, 5.2, 1.7],
            ]),
            batch_X[1]
        ))
        self.assertTrue(np.array_equal(
            np.array([
                [-2.4, 4.1, 5.2, 1.7],
                [2.0, 5.0, 4.0, 1.9],
            ]),
            batch_Y[1]
        ))


if __name__ == '__main__':
    unittest.main(verbosity=2)
