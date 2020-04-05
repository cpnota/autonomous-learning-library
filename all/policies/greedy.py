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

    def __call__(self, state):
        if np.random.rand() < self.epsilon:
            return torch.randint(self.num_actions, (len(state),), device=self.q.device)
        return torch.argmax(self.q(state), dim=1)

    def no_grad(self, state):
        if np.random.rand() < self.epsilon:
            return torch.randint(self.num_actions, (len(state),), device=self.q.device)
        return torch.argmax(self.q.no_grad(state), dim=1)

    def eval(self, state):
        return torch.argmax(self.q.eval(state), dim=1)
