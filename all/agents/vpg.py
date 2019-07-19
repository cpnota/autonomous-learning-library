import torch
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
        self._rewards = []

    def act(self, state, reward):
        if not self._features:
            return self._initial(state)
        if not state.done:
            return self._act(state, reward)
        return self._terminal(reward)

    def _initial(self, state):
        features = self.features(state)
        self._features = [features.features]
        return self.policy(features)

    def _act(self, state, reward):
        features = self.features(state)
        self._features.append(features.features)
        self._rewards.append(reward)
        return self.policy(features)

    def _terminal(self, reward):
        self._rewards.append(reward)
        features = torch.cat(self._features)
        rewards = torch.tensor(self._rewards, device=features.device)
        self._trajectories.append((features, rewards))
        self._current_batch_size += len(features)
        self._features = []
        self._rewards = []

        if self._current_batch_size >= self.min_batch_size:
            self._train()

    def _train(self):
        advantages = torch.cat([
            self._compute_advantages(features, rewards)
            for (features, rewards)
            in self._trajectories
        ])
        self.v.reinforce(advantages, retain_graph=True)
        self.policy.reinforce(advantages)
        self.features.reinforce()
        self._trajectories = []
        self._current_batch_size = 0

    def _compute_advantages(self, features, rewards):
        returns = self._compute_discounted_returns(rewards)
        values = self.v(State(features))
        return returns - values

    def _compute_discounted_returns(self, rewards):
        returns = rewards.clone()
        t = len(returns) - 1
        discounted_return = 0
        for reward in torch.flip(rewards, dims=(0,)):
            discounted_return = reward + self.gamma * discounted_return
            returns[t] = discounted_return
            t -= 1
        return returns
