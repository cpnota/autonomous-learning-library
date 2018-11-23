import unittest
from gym.spaces import Discrete
import numpy as np
from . import TabularStateValue

LEARNING_RATE = 0.1
STATE_SPACE = Discrete(3)
STATE = 2


class TabularStateValueTest(unittest.TestCase):
    def setUp(self):
        self.approximation = TabularStateValue(LEARNING_RATE, STATE_SPACE)

    def test_call_initial(self):
        np.testing.assert_equal(self.approximation.call(STATE), 0)

    def test_update(self):
        np.testing.assert_equal(self.approximation.call(STATE), 0)
        self.approximation.update(1, STATE)
        np.testing.assert_equal(self.approximation.call(STATE), np.array(LEARNING_RATE))

    def test_gradient(self):
        expected = np.zeros((STATE_SPACE.n))
        expected[STATE] = 1
        np.testing.assert_allclose(self.approximation.gradient(STATE), expected)

    def test_get_parameters(self):
        np.testing.assert_equal(
            self.approximation.parameters,
            np.zeros((STATE_SPACE.n))
        )

    def test_set_parameters(self):
        new_parameters = np.ones((STATE_SPACE.n))
        self.approximation.parameters = new_parameters
        np.testing.assert_equal(
            self.approximation.parameters,
            new_parameters
        )

    def test_update_parameters(self):
        gradient = np.ones((STATE_SPACE.n))
        self.approximation.apply(gradient)
        np.testing.assert_equal(
            self.approximation.parameters,
            LEARNING_RATE * gradient
        )


if __name__ == '__main__':
    unittest.main()
