from torch.nn.functional import mse_loss
from ._agent import Agent


class VAC(Agent):
    '''Vanilla Actor-Critic'''
    def __init__(self, features, v, policy, gamma=1):
        self.features = features
        self.v = v
        self.policy = policy
        self.gamma = gamma
        self._features = None
        self._distribution = None
        self._action = None

    def act(self, state, reward):
        self._train(state, reward)
        self._features = self.features(state)
        self._distribution = self.policy(self._features)
        self._action = self._distribution.sample()
        return self._action

    def _train(self, state, reward):
        if self._features:
            # forward pass
            values = self.v(self._features)
            targets = reward + self.gamma * self.v.target(self.features.target(state))
            advantages = targets - values.detach()
            # compute losses
            value_loss = mse_loss(values, targets)
            policy_loss = -(advantages * self._distribution.log_prob(self._action)).mean()
            # backward pass
            self.v.reinforce(value_loss)
            self.policy.reinforce(policy_loss)
            self.features.reinforce()
