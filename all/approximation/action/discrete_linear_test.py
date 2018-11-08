from all.approximation.action.discrete_linear import DiscreteLinearApproximation
from all.approximation.bases.fourier import FourierBasis
from gym.spaces import Box
import numpy as np
import unittest

num_actions = 3
learning_rate = 0.1
space = Box(low=0, high=1, shape=(2,))
basis = FourierBasis(space, 2, 2)
state = np.array([0.5, 1])

class TestDiscreteLinearApproximation(unittest.TestCase):
  def setUp(self):
    self.approximation = DiscreteLinearApproximation(learning_rate, basis, actions=num_actions)

  def test_call_initial(self):
    np.testing.assert_equal(self.approximation.call(state), np.array([0, 0, 0]))

  def test_update(self):
    np.testing.assert_equal(self.approximation.call(state), np.array([0, 0, 0]))
    self.approximation.update(1, state, 1)
    np.testing.assert_allclose(self.approximation.call(state), np.array([0, 0.6, 0]))

  def test_call_single(self):
    np.testing.assert_equal(self.approximation.call(state), np.array([0, 0, 0]))
    self.approximation.update(1, state, 1)
    np.testing.assert_approx_equal(self.approximation.call(state, 1), 0.6)

  def test_get_parameters(self):
    np.testing.assert_equal(
      self.approximation.get_parameters(), 
      np.zeros((num_actions, basis.num_features))
    )

  def test_set_parameters(self):
    new_parameters = np.ones((num_actions, basis.num_features))
    self.approximation.set_parameters(new_parameters)
    np.testing.assert_equal(
      self.approximation.get_parameters(),
      new_parameters
    )

  def test_update_parameters(self):
    gradient = np.ones((num_actions, basis.num_features))
    self.approximation.update_parameters(gradient)
    np.testing.assert_equal(
      self.approximation.get_parameters(),
      learning_rate * gradient
    )

if __name__ == '__main__':
    unittest.main()
