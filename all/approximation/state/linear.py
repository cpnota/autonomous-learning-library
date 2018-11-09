import numpy as np


class LinearApproximation:
    def __init__(self, alpha, basis):
        self.alpha = alpha
        self.basis = basis
        self.weights = np.zeros(self.basis.num_features)

    def call(self, state):
        features = self.basis.features(state)
        return self.weights.dot(features)

    def update(self, error, state):
        features = self.basis.features(state)
        self.weights += self.alpha * error * features

    def gradient(self, state):
        return self.basis.features(state)

    def get_parameters(self):
        return self.weights

    def set_parameters(self, weights):
        self.weights = weights

    def update_parameters(self, errors):
        self.weights += self.alpha * errors
