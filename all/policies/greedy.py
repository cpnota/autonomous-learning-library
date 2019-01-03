import numpy as np
from all.policies.policy import Policy

class Greedy(Policy):
    def __init__(self, q, epsilon=0.1):
        self.q = q
        self.epsilon = epsilon

    def __call__(self, state, action=None, prob=False):
        action_scores = self.q(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(action_scores.shape[0])
        best = np.argwhere(action_scores == np.max(action_scores)).flatten()
        return np.random.choice(best)

    def update(self, error, state, action):
        return self.q.update(error, state, action)

    def gradient(self, state, action):
        return self.q.gradient(state, action)

    def apply(self, gradient):
        return self.q.apply(gradient)

    @property
    def parameters(self):
        return self.q.parameters

    @parameters.setter
    def parameters(self, parameters):
        self.q.parameters = parameters
