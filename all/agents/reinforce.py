import torch
from .abstract import Agent

# pylint: disable=W0201


class REINFORCE(Agent):
    def __init__(self, v, policy):
        self.v = v
        self.policy = policy

    def new_episode(self, env):
        self.env = env
        self.next_state = self.env.state
        self.states = []
        self.values = []
        self.values = []
        self.rewards = []

    def act(self):
        state = self.env.state
        action = self.policy(state)
        self.env.step(action)

        self.states.append(state)
        self.rewards.append(self.env.reward)

        if self.env.done:
            self.update()

    def update(self):
        states = torch.cat(self.states)
        rewards = torch.tensor(self.rewards)

        values = self.v(states)
        ordered = torch.flip(rewards, dims=(0,))
        returns = torch.flip(torch.cumsum(ordered, dim=0), dims=(0,))
        advantages = returns - values

        self.v.reinforce(advantages)
        self.policy.reinforce(advantages)
