import torch
from .abstract import Agent

# pylint: disable=W0201
class REINFORCE(Agent):
    def __init__(self, v, policy):
        self.v = v
        self.policy = policy

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

        values = self.v(states)
        ordered = torch.flip(rewards, dims=(0,))
        returns = torch.flip(torch.cumsum(ordered, dim=0), dims=(0,))
        advantages = returns - values

        self.v.reinforce(advantages)
        self.policy.reinforce(advantages)
