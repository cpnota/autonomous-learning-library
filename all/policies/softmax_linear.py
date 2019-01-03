import numpy as np
from all.policies.policy import Policy

class SoftmaxLinear(Policy):
    def __init__(self, learning_rate, basis, actions):
        self.learning_rate = learning_rate
        self.basis = basis
        self.weights = np.zeros((actions, self.basis.num_features))

    def __call__(self, state, action=None, prob=False):
        features = self.basis.features(state)
        probabilities = self.probabilities(features)
        return np.random.choice(probabilities.shape[0], p=probabilities)

    def update(self, error, state, action):
        self.weights += self.learning_rate * error * self.gradient(state, action)

    def gradient(self, state, action):
        features = self.basis.features(state)
        neg_probabilities = -self.probabilities(features)
        neg_probabilities[action] += 1
        return np.outer(neg_probabilities, features)

    def apply(self, gradient):
        self.weights += self.learning_rate * gradient

    @property
    def parameters(self):
        return self.weights

    @parameters.setter
    def parameters(self, parameters):
        self.weights = parameters

    def probabilities(self, features):
        action_scores = np.exp(self.weights.dot(features))
        return action_scores / np.sum(action_scores)
