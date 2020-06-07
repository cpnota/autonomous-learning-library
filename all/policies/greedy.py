import numpy as np
import torch
from all.optim import Schedulable

class GreedyPolicy(Schedulable):
    '''
    An  "epsilon-greedy" action selection policy for discrete action spaces.

    This policy will usually choose the optimal action according to an approximation
    of the action value function (the "q-function"), but with probabilty epsilon will
    choose a random action instead. GreedyPolicy is a Schedulable, meaning that
    epsilon can be varied over time by passing a Scheduler object.

    Args:
        q (all.approximation.QNetwork): The action-value or "q-function"
        num_actions (int): The number of available actions.
        epsilon (float, optional): The probability of selecting a random action.
    '''
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
