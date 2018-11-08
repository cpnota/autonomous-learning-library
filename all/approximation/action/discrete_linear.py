import numpy as np

class DiscreteLinearApproximation:
  def __init__(self, alpha, basis, actions):
    self.alpha = alpha
    self.basis = basis
    self.weights = np.zeros((actions, self.basis.num_features))

  def call(self, state, action=None):
    features = self.basis.features(state)
    if (action == None):
      return self.weights.dot(features)
    return self.weights[action].dot(features)
  
  def update(self, error, state, action):
    features = self.basis.features(state)
    self.weights[action] += self.alpha * error * features

  def gradient(self, state, action):
    grad = np.zeros(self.weights.shape)
    grad[action] = self.basis.features(state)
    return grad

  def get_parameters(self):
    return self.weights

  def set_parameters(self, parameters):
    self.weights = parameters
    return self

  def update_parameters(self, gradient):
    self.weights += self.alpha * gradient
    return self
