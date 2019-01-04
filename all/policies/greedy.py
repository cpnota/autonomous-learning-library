import numpy as np
import torch
from .abstract import Policy

class GreedyPolicy(Policy):
    def __init__(self, q, epsilon=0.1):
        self.q = q
        self.epsilon = epsilon

    def __call__(self, state, action=None, prob=False):
        action_scores = self.q(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(action_scores.shape[0])
        return torch.argmax(action_scores).item()

    def update(self, error, state, action):
        return self.q.update(error, state, action)
