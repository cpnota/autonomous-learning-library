import numpy as np
from all.approximation.value.action.action_value_approximation import ActionValueApproximation

class LinearStateDiscreteActionValue(ActionValueApproximation):
    def __init__(self, alpha, basis, actions):
        self.alpha = alpha
        self.basis = basis
        self.weights = np.zeros((actions, self.basis.num_features))

    def __call__(self, state, action=None):
        features = self.basis.features(state)
        if action is None:
            return self.weights.dot(features)
        return self.weights[action].dot(features)

    def update(self, error, state, action):
        features = self.basis.features(state)
        self.weights[action] += self.alpha * error * features

    def gradient(self, state, action):
        grad = np.zeros(self.weights.shape)
        grad[action] = self.basis.features(state)
        return grad

    def apply(self, gradient):
        self.weights += self.alpha * gradient
        return self

    @property
    def parameters(self):
        return self.weights

    @parameters.setter
    def parameters(self, weights):
        self.weights = weights
