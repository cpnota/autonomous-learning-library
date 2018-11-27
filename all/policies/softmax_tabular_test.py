import unittest
from gym.spaces import Discrete
import numpy as np
from all.policies import SoftmaxTabular

LEARNING_RATE = 0.1
STATE_SPACE = Discrete(5)
ACTION_SPACE = Discrete(3)
STATE = 2
ACTION = 1


class TestSoftmaxTabular(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.policy = SoftmaxTabular(LEARNING_RATE, STATE_SPACE, ACTION_SPACE)

    def test_call_initial(self):
        np.testing.assert_equal(self.policy.call(STATE), 1)

    def test_initial_probabilities(self):
        np.testing.assert_allclose(
            self.policy.probabilities(STATE),
            np.array([1/3, 1/3, 1/3])
        )

    def test_update(self):
        self.policy.update(1, STATE, ACTION)
        np.testing.assert_array_almost_equal(
            self.policy.probabilities(STATE),
            np.array([0.322043, 0.355913, 0.322043]))

    def test_gradient(self):
        features = np.zeros((STATE_SPACE.n))
        features[STATE] = 1
        np.testing.assert_allclose(
            self.policy.gradient(STATE, ACTION),
            np.array(np.array([
                -features / 3,
                features * 2/3,
                -features / 3
            ]).T)
        )

    def test_get_parameters(self):
        np.testing.assert_equal(
            self.policy.parameters,
            np.zeros((STATE_SPACE.n, ACTION_SPACE.n))
        )

    def test_set_parameters(self):
        new_parameters = np.ones((STATE_SPACE.n, ACTION_SPACE.n))
        self.policy.parameters = new_parameters
        np.testing.assert_equal(
            self.policy.parameters,
            new_parameters
        )

    def test_update_parameters(self):
        gradient = np.ones((STATE_SPACE.n, ACTION_SPACE.n))
        self.policy.apply(gradient)
        np.testing.assert_equal(
            self.policy.parameters,
            LEARNING_RATE * gradient
        )


if __name__ == '__main__':
    unittest.main()
