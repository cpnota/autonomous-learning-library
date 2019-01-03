import unittest
import numpy as np
from gym.spaces import Box
from all.approximation.value.action import LinearStateDiscreteActionValue
from all.approximation.bases.fourier import FourierBasis
from all.policies.greedy import Greedy


SPACE = Box(low=0, high=1, shape=(2,), dtype=np.float32)


class TestGreedyPolicy(unittest.TestCase):
    def test_choose_greedy(self):
        basis = FourierBasis(SPACE, 2)
        approximation = LinearStateDiscreteActionValue(0.1, basis, actions=3)
        policy = Greedy(approximation, 0)
        state = np.array([0.5, 1])

        approximation.update(1, state, 1)
        self.assertEqual((policy(state)), 1)


if __name__ == '__main__':
    unittest.main()
