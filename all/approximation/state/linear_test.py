from approximation.state.linear import LinearApproximation
from approximation.bases.fourier import FourierBasis
import numpy as np
import unittest

class TestLinearFunctionApproximation(unittest.TestCase):
  def test_init(self):
    basis = FourierBasis(2, 2, 2)
    approximation = LinearApproximation(0.1, basis)
    x = np.array([0.5, 1])
    self.assertEqual(approximation.call(x), 0)

  def test_update(self):
    basis = FourierBasis(2, 2, 2)
    approximation = LinearApproximation(0.1, basis)
    x = np.array([0.5, 1])
    self.assertEqual(approximation.call(x), 0)
    approximation.update(x, 1)
    self.assertAlmostEqual(approximation.call(x), 0.6)

if __name__ == '__main__':
    unittest.main()
