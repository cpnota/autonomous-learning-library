import unittest
from gym.spaces import Box
import numpy as np
from all.approximation.STATE import LinearApproximation
from all.approximation.bases import FourierBasis

NUM_ACTIONS = 3
LEARNING_RATE = 0.1
SPACE = Box(low=0, high=1, shape=(2,))
BASIS = FourierBasis(SPACE, 2)
STATE = np.array([0.5, 1])


class TestLinearFunctionApproximation(unittest.TestCase):
    def setUp(self):
        self.approximation = LinearApproximation(LEARNING_RATE, BASIS
                                                 )

    def test_call_init(self):
        self.assertEqual(self.approximation.call(STATE), 0)

    def test_update(self):
        self.assertEqual(self.approximation.call(STATE), 0)
        self.approximation.update(1, STATE)
        self.assertAlmostEqual(self.approximation.call(STATE), 0.6)

    def test_gradient(self):
        np.testing.assert_equal(
            self.approximation.gradient(STATE),
            BASIS.features(STATE)
        )

    def test_get_parameters(self):
        np.testing.assert_equal(
            self.approximation.get_parameters(),
            np.zeros((BASIS.num_features))
        )

    def test_set_parameters(self):
        new_parameters = np.ones((BASIS.num_features))
        self.approximation.set_parameters(new_parameters)
        np.testing.assert_equal(
            self.approximation.get_parameters(),
            new_parameters
        )

    def test_update_parameters(self):
        gradient = np.ones((BASIS.num_features))
        self.approximation.update_parameters(gradient)
        np.testing.assert_equal(
            self.approximation.get_parameters(),
            LEARNING_RATE
            * gradient
        )


if __name__ == '__main__':
    unittest.main()
