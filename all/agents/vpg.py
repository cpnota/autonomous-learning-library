import torch
from torch.nn.functional import mse_loss
from all.environments import State
from ._agent import Agent

class VPG(Agent):
    '''Vanilla Policy Gradient'''
    def __init__(
            self,
            features,
            v,
            policy,
            gamma=0.99,
            # run complete episodes until we have
            # seen at least min_batch_size states
            min_batch_size=1
    ):
        self.features = features
        self.v = v
        self.policy = policy
        self.gamma = gamma
        self.min_batch_size = min_batch_size
        self._current_batch_size = 0
        self._trajectories = []
        self._features = []
        self._log_pis = []
        self._rewards = []

    def act(self, state, reward):
        if not self._features:
            return self._initial(state)
        if not state.done:
            return self._act(state, reward)
        return self._terminal(state, reward)

    def _initial(self, state):
        features = self.features(state)
        distribution = self.policy(features)
        action = distribution.sample()
        self._features = [features.features]
        self._log_pis.append(distribution.log_prob(action))
        return action

    def _act(self, state, reward):
        features = self.features(state)
        distribution = self.policy(features)
        action = distribution.sample()
        self._features.append(features.features)
        self._rewards.append(reward)
        self._log_pis.append(distribution.log_prob(action))
        return action

    def _terminal(self, state, reward):
        self._rewards.append(reward)
        features = torch.cat(self._features)
        rewards = torch.tensor(self._rewards, device=features.device)
        log_pis = torch.cat(self._log_pis)
        self._trajectories.append((features, rewards, log_pis))
        self._current_batch_size += len(features)
        self._features = []
        self._rewards = []
        self._log_pis = []

        if self._current_batch_size >= self.min_batch_size:
            self._train()

        # have to return something
        return self.policy.eval(self.features.eval(state)).sample()

    def _train(self):
        # forward pass
        values = torch.cat([
            self.v(State(features))
            for (features, _, _)
            in self._trajectories
        ])
        targets = torch.cat([
            self._compute_discounted_returns(rewards)
            for (_, rewards, _)
            in self._trajectories
        ])
        log_pis = torch.cat([log_pis for (_, _, log_pis) in self._trajectories])
        advantages = targets - values.detach()
        # compute losses
        value_loss = mse_loss(values, targets)
        policy_loss = -(advantages * log_pis).mean()
        # backward pass
        self.v.reinforce(value_loss)
        self.policy.reinforce(policy_loss)
        self.features.reinforce()
        # cleanups
        self._trajectories = []
        self._current_batch_size = 0

    def _compute_discounted_returns(self, rewards):
        returns = rewards.clone()
        t = len(returns) - 1
        discounted_return = 0
        for reward in torch.flip(rewards, dims=(0,)):
            discounted_return = reward + self.gamma * discounted_return
            returns[t] = discounted_return
            t -= 1
        return returns
