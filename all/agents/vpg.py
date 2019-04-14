import torch
from .abstract import Agent

class VPG(Agent):
    def __init__(
            self,
            features,
            v,
            policy,
            gamma=0.99,
            n_episodes=1
    ):
        self.features = features
        self.v = v
        self.policy = policy
        self.gamma = gamma
        self.n_episodes = n_episodes
        self._trajectories = []
        self._states = None
        self._rewards = None

    def initial(self, state, info=None):
        features = self.features(state)
        self._states = [features]
        self._rewards = []
        return self.policy(features)

    def act(self, state, reward, info=None):
        features = self.features(state)
        self._states.append(features)
        self._rewards.append(reward)
        return self.policy(features)

    def terminal(self, reward, info=None):
        self._rewards.append(reward)
        states = torch.cat(self._states)
        rewards = torch.tensor(self._rewards, device=states.device)
        self._trajectories.append((states, rewards))
        if len(self._trajectories) >= self.n_episodes:
            advantages = torch.cat([
                self._compute_advantages(states, rewards)
                for (states, rewards)
                in self._trajectories
            ])
            self.v.reinforce(advantages, retain_graph=True)
            self.policy.reinforce(advantages)
            self.features.reinforce()
            self._trajectories = []

    def _compute_advantages(self, features, rewards):
        returns = self._compute_discounted_returns(rewards)
        values = self.v(features)
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
