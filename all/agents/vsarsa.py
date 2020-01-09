from torch.nn.functional import mse_loss
from ._agent import Agent


class VSarsa(Agent):
    '''Vanilla SARSA'''
    def __init__(self, q, policy, gamma=1):
        self.q = q
        self.policy = policy
        self.gamma = gamma
        self._state = None
        self._action = None

    def act(self, state, reward):
        action = self.policy(state)
        self._train(reward, state, action)
        self._state = state
        self._action = action
        return action

    def _train(self, reward, next_state, next_action):
        if self._state:
            # forward pass
            value = self.q(self._state, self._action)
            # compute target
            target = reward + self.gamma * self.q.target(next_state, next_action)
            # compute loss
            loss = mse_loss(value, target)
            # backward pass
            self.q.reinforce(loss)
