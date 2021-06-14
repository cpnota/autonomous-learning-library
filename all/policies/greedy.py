import numpy as np
import torch
from all.optim import Schedulable


class GreedyPolicy(Schedulable):
    '''
    An  "epsilon-greedy" action selection policy for discrete action spaces.

    This policy will usually choose the optimal action according to an approximation
    of the action value function (the "q-function"), but with probability epsilon will
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
            return np.random.randint(0, self.num_actions)
        return torch.argmax(self.q(state)).item()

    def no_grad(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        return torch.argmax(self.q.no_grad(state)).item()

    def eval(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        return torch.argmax(self.q.eval(state)).item()


class ParallelGreedyPolicy(Schedulable):
    '''
    A parallel version of the "epsilon-greedy" action selection policy for discrete action spaces.

    This policy will usually choose the optimal action according to an approximation
    of the action value function (the "q-function"), but with probability epsilon will
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
        return self._choose_action(self.q(state))

    def no_grad(self, state):
        return self._choose_action(self.q.no_grad(state))

    def eval(self, state):
        return self._choose_action(self.q.eval(state))

    def _choose_action(self, action_values):
        best_actions = torch.argmax(action_values, dim=-1)
        random_actions = torch.randint(0, self.num_actions, best_actions.shape, device=best_actions.device)
        choices = (torch.rand(best_actions.shape, device=best_actions.device) < self.epsilon).int()
        return choices * random_actions + (1 - choices) * best_actions
