import numpy as np
from all.policies.policy import Policy

class SoftmaxTabular(Policy):
    def __init__(self, learning_rate, state_space, action_space):
        self.learning_rate = learning_rate
        self.weights = np.zeros((state_space.n, action_space.n))

    def call(self, state, action=None, prob=False):
        probabilities = self.probabilities(state)
        return np.random.choice(probabilities.shape[0], p=probabilities)

    def update(self, error, state, action):
        self.weights[state] += self.learning_rate * error * self.gradient_for_state(state, action)

    def gradient(self, state, action):
        grad = np.zeros((self.weights.shape))
        neg_probabilities = self.gradient_for_state(state, action)
        grad[state] = neg_probabilities
        return grad

    def apply(self, gradient):
        self.weights += self.learning_rate * gradient

    @property
    def parameters(self):
        return self.weights

    @parameters.setter
    def parameters(self, parameters):
        self.weights = parameters

    def probabilities(self, state):
        action_scores = np.exp(self.weights[state])
        return action_scores / np.sum(action_scores)

    def gradient_for_state(self, state, action):
        """
        Gradient over the actions given a state.
        This should just be a single vector.
        """
        neg_probabilities = -self.probabilities(state)
        neg_probabilities[action] += 1
        return neg_probabilities
