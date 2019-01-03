import unittest
from gym.spaces import Box
import numpy as np
from all.policies import SoftmaxLinear
from all.approximation.bases.fourier import FourierBasis

NUM_ACTIONS = 3
LEARNING_RATE = 0.1
SPACE = Box(low=0, high=1, shape=(2,), dtype=np.float32)
BASIS = FourierBasis(SPACE, 2)
STATE = np.array([0.5, 1])
FEATURES = BASIS.features(STATE)


class TestLinearStateDiscreteActionValue(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.policy = SoftmaxLinear(
            LEARNING_RATE, BASIS, actions=NUM_ACTIONS)

    def test_call_initial(self):
        np.testing.assert_equal(
            self.policy(STATE), 1)

    def test_update(self):
        self.policy.update(1, STATE, 0)
        np.testing.assert_allclose(
            self.policy.probabilities(FEATURES),
            np.array([0.47673, 0.261635, 0.261635]
                     ))

    def test_gradient(self):
        np.testing.assert_allclose(
            self.policy.gradient(STATE, 1),
            np.array([
                -FEATURES / 3,
                FEATURES * 2/3,
                -FEATURES / 3
            ])
        )

    def test_get_parameters(self):
        np.testing.assert_equal(
            self.policy.parameters,
            np.zeros((NUM_ACTIONS, BASIS.num_features))
        )

    def test_set_parameters(self):
        new_parameters = np.ones((NUM_ACTIONS, BASIS.num_features))
        self.policy.parameters = new_parameters
        np.testing.assert_equal(
            self.policy.parameters,
            new_parameters
        )

    def test_update_parameters(self):
        gradient = np.ones((NUM_ACTIONS, BASIS
                            .num_features))
        self.policy.apply(gradient)
        np.testing.assert_equal(
            self.policy.parameters,
            LEARNING_RATE * gradient
        )


if __name__ == '__main__':
    unittest.main()
