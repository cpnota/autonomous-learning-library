from all.approximation.action.discrete_linear import DiscreteLinearApproximation
from all.approximation.bases.fourier import FourierBasis
from gym.spaces import Box
import numpy as np
import unittest

space = Box(low=0, high=1, shape=(2,))

class TestDiscreteLinearApproximation(unittest.TestCase):
  def test_init(self):
    basis = FourierBasis(space, 2, 2)
    approximation = DiscreteLinearApproximation(0.1, basis, actions=3)
    x = np.array([0.5, 1])
    np.testing.assert_equal(approximation.call(x), np.array([0, 0, 0]))

  def test_update(self):
    basis = FourierBasis(space, 2, 2)
    approximation = DiscreteLinearApproximation(0.1, basis, actions=3)
    x = np.array([0.5, 1])
    np.testing.assert_equal(approximation.call(x), np.array([0, 0, 0]))
    approximation.update(x, 1, 1)
    np.testing.assert_allclose(approximation.call(x), np.array([0, 0.6, 0]))

  def test_call_one(self):
    basis = FourierBasis(space, 2, 2)
    approximation = DiscreteLinearApproximation(0.1, basis, actions=3)
    x = np.array([0.5, 1])
    np.testing.assert_equal(approximation.call(x), np.array([0, 0, 0]))
    approximation.update(x, 1, 1)
    np.testing.assert_approx_equal(approximation.call(x, 1), 0.6)

if __name__ == '__main__':
    unittest.main()
