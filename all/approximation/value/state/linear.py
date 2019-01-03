import numpy as np
from all.approximation.value.state.state_value_approximation import StateValueApproximation


class LinearStateValue(StateValueApproximation):
    def __init__(self, alpha, basis):
        self.alpha = alpha
        self.basis = basis
        self.weights = np.zeros(self.basis.num_features)

    def __call__(self, state):
        if state is None:
            return 0
        features = self.basis.features(state)
        return self.weights.dot(features)

    def update(self, error, state):
        features = self.basis.features(state)
        self.weights += self.alpha * error * features

    def gradient(self, state):
        return self.basis.features(state)

    def apply(self, gradient):
        self.weights += self.alpha * gradient

    @property
    def parameters(self):
        return self.weights

    @parameters.setter
    def parameters(self, weights):
        self.weights = weights
