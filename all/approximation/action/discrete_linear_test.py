import unittest
from gym.spaces import Box
import numpy as np
from all.approximation.action.discrete_linear import DiscreteLinearApproximation
from all.approximation.bases.fourier import FourierBasis

NUM_ACTIONS = 3
LEARNING_RATE = 0.1
SPACE = Box(low=0, high=1, shape=(2,))
BASIS = FourierBasis(SPACE, 2)
STATE = np.array([0.5, 1])


class TestDiscreteLinearApproximation(unittest.TestCase):
    def setUp(self):
        self.approximation = DiscreteLinearApproximation(
            LEARNING_RATE, BASIS, actions=NUM_ACTIONS)

    def test_call_initial(self):
        np.testing.assert_equal(
            self.approximation.call(STATE), np.array([0, 0, 0]))

    def test_update(self):
        np.testing.assert_equal(
            self.approximation.call(STATE), np.array([0, 0, 0]))
        self.approximation.update(1, STATE, 1)
        np.testing.assert_allclose(
            self.approximation.call(STATE), np.array([0, 0.6, 0]))

    def test_call_single(self):
        np.testing.assert_equal(
            self.approximation.call(STATE), np.array([0, 0, 0]))
        self.approximation.update(1, STATE, 1)
        np.testing.assert_approx_equal(
            self.approximation.call(STATE, 1), 0.6)

    def test_gradient(self):
        features = BASIS.features(STATE)
        expected = np.array([
            np.zeros(features.shape[0]),
            features,
            np.zeros(features.shape[0])
        ])
        np.testing.assert_allclose(
            self.approximation.gradient(STATE, 1), expected)

    def test_get_parameters(self):
        np.testing.assert_equal(
            self.approximation.get_parameters(),
            np.zeros((NUM_ACTIONS, BASIS
                      .num_features))
        )

    def test_set_parameters(self):
        new_parameters = np.ones((NUM_ACTIONS, BASIS
                                  .num_features))
        self.approximation.set_parameters(new_parameters)
        np.testing.assert_equal(
            self.approximation.get_parameters(),
            new_parameters
        )

    def test_update_parameters(self):
        gradient = np.ones((NUM_ACTIONS, BASIS
                            .num_features))
        self.approximation.update_parameters(gradient)
        np.testing.assert_equal(
            self.approximation.get_parameters(),
            LEARNING_RATE * gradient
        )


if __name__ == '__main__':
    unittest.main()
