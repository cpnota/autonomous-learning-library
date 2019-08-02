import numpy as np
import torch
from all.optim import Schedulable

class GreedyPolicy(Schedulable):
    def __init__(
            self,
            q,
            num_actions,
            epsilon=0.,
    ):
        self.q = q
        self.num_actions = num_actions
        self.epsilon = epsilon

    def __call__(self, state, action=None, prob=False):
        if np.random.rand() < self.epsilon:
            return torch.randint(self.num_actions, (len(state),), device=self.q.device)
        with torch.no_grad():
            action_scores = self.q.eval(state)
        return torch.argmax(action_scores, dim=1)
