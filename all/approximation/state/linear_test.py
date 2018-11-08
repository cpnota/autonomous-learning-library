from all.approximation.state import LinearApproximation
from all.approximation.bases import FourierBasis
from gym.spaces import Box
import numpy as np
import unittest

space = Box(low=0, high=1, shape=(2,))

class TestLinearFunctionApproximation(unittest.TestCase):
  def test_call_init(self):
    basis = FourierBasis(space, 2, 2)
    approximation = LinearApproximation(0.1, basis)
    x = np.array([0.5, 1])
    self.assertEqual(approximation.call(x), 0)

  def test_update(self):
    basis = FourierBasis(space, 2, 2)
    approximation = LinearApproximation(0.1, basis)
    x = np.array([0.5, 1])
    self.assertEqual(approximation.call(x), 0)
    approximation.update(1, x)
    self.assertAlmostEqual(approximation.call(x), 0.6)

  def test_gradient(self):
    basis = FourierBasis(space, 2, 2)
    approximation = LinearApproximation(0.1, basis)
    x = np.array([0.5, 1])
    np.testing.assert_equal(
      approximation.gradient(x),
      basis.features(x)
    )

if __name__ == '__main__':
    unittest.main()
