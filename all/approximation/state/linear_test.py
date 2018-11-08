from all.approximation.state import LinearApproximation
from all.approximation.bases import FourierBasis
from gym.spaces import Box
import numpy as np
import unittest

num_actions = 3
learning_rate = 0.1
space = Box(low=0, high=1, shape=(2,))
basis = FourierBasis(space, 2, 2)
state = np.array([0.5, 1])

class TestLinearFunctionApproximation(unittest.TestCase):
  def setUp(self):
    self.approximation = LinearApproximation(learning_rate, basis)

  def test_call_init(self):
    self.assertEqual(self.approximation.call(state), 0)

  def test_update(self):
    self.assertEqual(self.approximation.call(state), 0)
    self.approximation.update(1, state)
    self.assertAlmostEqual(self.approximation.call(state), 0.6)

  def test_gradient(self):
    np.testing.assert_equal(
      self.approximation.gradient(state),
      basis.features(state)
    )

  def test_get_parameters(self):
    np.testing.assert_equal(
      self.approximation.get_parameters(), 
      np.zeros((basis.num_features))
    )

  def test_set_parameters(self):
    new_parameters = np.ones((basis.num_features))
    self.approximation.set_parameters(new_parameters)
    np.testing.assert_equal(
      self.approximation.get_parameters(),
      new_parameters
    )

  def test_update_parameters(self):
    gradient = np.ones((basis.num_features))
    self.approximation.update_parameters(gradient)
    np.testing.assert_equal(
      self.approximation.get_parameters(),
      learning_rate * gradient
    )

if __name__ == '__main__':
    unittest.main()
