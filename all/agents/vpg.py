import torch
from .abstract import Agent

# pylint: disable=W0201
class VPG(Agent):
    def __init__(
            self,
            v,
            policy,
            gamma=0.99,
            n_episodes=1
    ):
        self.v = v
        self.policy = policy
        self.gamma = gamma
        self.n_episodes = n_episodes
        self._trajectories = []

    def initial(self, state, info=None):
        self.states = [state]
        self.rewards = []
        return self.policy(state)

    def act(self, state, reward, info=None):
        self.states.append(state)
        self.rewards.append(reward)
        return self.policy(state)

    def terminal(self, reward, info=None):
        self.rewards.append(reward)
        states = torch.cat(self.states)
        rewards = torch.tensor(self.rewards, device=states.device)
        self._trajectories.append((states, rewards))
        if len(self._trajectories) >= self.n_episodes:
            advantages = torch.cat([
                self._compute_advantages(states, rewards)
                for (states, rewards)
                in self._trajectories
            ])
            self.v.reinforce(advantages)
            self.policy.reinforce(advantages)
            self._trajectories = []

    def _compute_advantages(self, states, rewards):
        returns = self._compute_discounted_returns(rewards)
        values = self.v(states)
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
