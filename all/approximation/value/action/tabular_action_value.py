import numpy as np
from all.approximation.value.action.action_value_approximation import ActionValueApproximation

class TabularActionValue(ActionValueApproximation):
    def __init__(self, alpha, state_space, action_space):
        self.alpha = alpha
        self.values = np.zeros((state_space.n, action_space.n))

    def __call__(self, state, action=None):
        if action is None:
            return self.values[state]
        return self.values[state, action]

    def update(self, error, state, action):
        self.values[state, action] += self.alpha * error

    def gradient(self, state, action):
        grad = np.zeros(self.values.shape)
        grad[state, action] = 1
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
