import numpy as np
import torch
from .abstract import Policy

class GreedyPolicy(Policy):
    def __init__(self, q, initial_epsilon=1., final_epsilon=0.1, annealing_time=1):
        self.q = q
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.executions = 0
        self.annealing_rate = (initial_epsilon - final_epsilon) / annealing_time
        self.annealing_time = annealing_time

    def __call__(self, state, action=None, prob=False):
        self.anneal()
        with torch.no_grad():
            action_scores = self.q(state).squeeze(0)
        if np.random.rand() < self.epsilon:
            return torch.tensor(np.random.randint(action_scores.shape[0]))

        # select randomly from the best
        # not sure how to do in pure torch
        scores = action_scores.detach().numpy()
        best = np.argwhere(scores == np.max(scores)).flatten()
        return torch.tensor(np.random.choice(best), dtype=torch.long)

    def reinforce(self, errors):
        return # not possible

    def anneal(self):
        self.executions += 1
        if self.executions > self.annealing_time:
            self.epsilon = self.final_epsilon
        else:
            self.epsilon -= self.annealing_rate
