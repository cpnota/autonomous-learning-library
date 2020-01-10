import torch
from torch.nn.functional import mse_loss
from ._agent import Agent


class VQN(Agent):
    '''Vanilla Q-Network'''
    def __init__(self, q, policy, discount_factor=1):
        self.q = q
        self.policy = policy
        self.discount_factor = discount_factor
        self._state = None
        self._action = None

    def act(self, state, reward):
        self._train(reward, state)
        action = self.policy(state)
        self._state = state
        self._action = action
        return action

    def _train(self, reward, next_state):
        if self._state:
            # forward pass
            value = self.q(self._state, self._action)
            # compute target
            target = reward + self.discount_factor * torch.max(self.q.target(next_state), dim=1)[0]
            # compute loss
            loss = mse_loss(value, target)
            # backward pass
            self.q.reinforce(loss)
