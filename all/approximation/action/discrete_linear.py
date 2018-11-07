import numpy as np

class DiscreteLinearApproximation:
  def __init__(self, alpha, basis, actions):
    self.alpha = alpha
    self.basis = basis
    self.weights = np.zeros((actions, self.basis.num_features))

  def call(self, state, action=None):
    features = self.basis.features(state)
    if (action == None):
      result = self.weights.dot(features)
      # print(result)
      return result
    return self.weights[action].dot(features)
  
  def update(self, state, action, error):
    features = self.basis.features(state)
    self.weights[action] += self.alpha * error * features
