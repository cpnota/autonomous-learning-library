from approximation.action.discrete_linear import DiscreteLinearApproximation
from approximation.bases.fourier import FourierBasis
from policies.greedy import Greedy
import numpy as np
import unittest

class TestGreedyPolicy(unittest.TestCase):
  def testChooseGreedy(self):
    basis = FourierBasis(2, 2, 2)
    approximation = DiscreteLinearApproximation(0.1, basis, actions=3)
    policy = Greedy(approximation, 0)
    state = np.array([0.5, 1])

    approximation.update(state, 1, 1)
    self.assertEqual((policy.choose_action(state)), 1)

if __name__ == '__main__':
    unittest.main()
