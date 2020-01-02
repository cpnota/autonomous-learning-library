import torch
from torch.nn.functional import mse_loss
from ._agent import Agent


class VQN(Agent):
    '''Vanilla Q-Network'''
    def __init__(self, q, policy, gamma=1):
        self.q = q
        self.policy = policy
        self.gamma = gamma
        self.env = None
        self.previous_state = None
        self.previous_action = None

    def act(self, state, reward):
        if self.previous_state:
            value = self.q(self.previous_state, self.previous_action)
            target = reward + self.gamma * torch.max(self.q.target(state), dim=1)[0]
            loss = mse_loss(value, target)
            self.q.reinforce(loss)
        action = self.policy(state)
        self.previous_state = state
        self.previous_action = action
        return action
