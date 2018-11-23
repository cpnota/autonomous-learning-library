import numpy as np
from .state_value_approximation import StateValueApproximation

class TabularStateValue(StateValueApproximation):
    def __init__(self, alpha, state_space):
        self.alpha = alpha
        self.values = np.zeros((state_space.n))

    def call(self, state):
        return self.values[state]

    def update(self, error, state):
        self.values[state] += self.alpha * error

    def gradient(self, state):
        grad = np.zeros(self.values.shape)
        grad[state] = 1
        return grad

    def apply(self, gradient):
        self.values += self.alpha * gradient
        return self

    @property
    def parameters(self):
        return self.values

    @parameters.setter
    def parameters(self, weights):
        self.values = weights
