import unittest
from gym.spaces import Discrete
import numpy as np
from . import TabularActionValue

NUM_ACTIONS = 3
LEARNING_RATE = 0.1
STATE_SPACE = Discrete(5)
ACTION_SPACE = Discrete(3)
STATE = 2
ACTION = 1


class TabularActionValueTest(unittest.TestCase):
    def setUp(self):
        self.approximation = TabularActionValue(
            LEARNING_RATE, STATE_SPACE, ACTION_SPACE)

    def test_call_initial(self):
        np.testing.assert_equal(self.approximation.call(STATE), np.array([0, 0, 0]))

    def test_update(self):
        np.testing.assert_equal(
            self.approximation.call(STATE), np.array([0, 0, 0]))
        self.approximation.update(1, STATE, ACTION)
        np.testing.assert_allclose(
            self.approximation.call(STATE), np.array([0, LEARNING_RATE, 0]))

    def test_call_single(self):
        np.testing.assert_equal(
            self.approximation.call(STATE), np.array([0, 0, 0]))
        self.approximation.update(1, STATE, ACTION)
        np.testing.assert_approx_equal(
            self.approximation.call(STATE, ACTION), LEARNING_RATE)

    def test_gradient(self):
        expected = np.zeros((STATE_SPACE.n, ACTION_SPACE.n))
        expected[STATE, ACTION] = 1
        np.testing.assert_allclose(
            self.approximation.gradient(STATE, ACTION), expected)

    def test_get_parameters(self):
        np.testing.assert_equal(
            self.approximation.parameters,
            np.zeros((STATE_SPACE.n, ACTION_SPACE.n))
        )

    def test_set_parameters(self):
        new_parameters = np.ones((STATE_SPACE.n, ACTION_SPACE.n))
        self.approximation.parameters = new_parameters
        np.testing.assert_equal(
            self.approximation.parameters,
            new_parameters
        )

    def test_update_parameters(self):
        gradient = np.ones((STATE_SPACE.n, ACTION_SPACE.n))
        self.approximation.apply(gradient)
        np.testing.assert_equal(
            self.approximation.parameters,
            LEARNING_RATE * gradient
        )


if __name__ == '__main__':
    unittest.main()
