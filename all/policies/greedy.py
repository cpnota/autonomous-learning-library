import numpy as np
import torch
from .abstract import Policy

class GreedyPolicy(Policy):
    def __init__(
            self,
            q,
            num_actions,
            initial_epsilon=1.,
            final_epsilon=0.1,
            annealing_start=0,
            annealing_time=1
    ):
        self.q = q
        self.num_actions = num_actions
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.executions = 0
        self.annealing_start = annealing_start
        self.annealing_end = annealing_start + annealing_time
        self.annealing_time = annealing_time

    def __call__(self, state, action=None, prob=False):
        self.epsilon = self.anneal()
        if np.random.rand() < self.epsilon:
            return torch.randint(self.num_actions, (1,), device=self.q.device)
        with torch.no_grad():
            action_scores = self.q(state).squeeze(0)
        return torch.argmax(action_scores)

    def reinforce(self, errors):
        return  # not possible

    def anneal(self):
        self.executions += 1
        if self.executions < self.annealing_start:
            return self.initial_epsilon
        if self.executions < self.annealing_end:
            alpha = (self.executions - self.annealing_start) / \
                (self.annealing_end - self.annealing_start)
            return (1 - alpha) * self.initial_epsilon + alpha * self.final_epsilon
        return self.final_epsilon
