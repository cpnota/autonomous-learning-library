import os
import torch
import gym
from all.agents import Agent

def watch(agent, env):
    action = None
    returns = 0
    while True:
        if env.done:
            print('returns:', returns)
            env.reset()
            returns = 0
        else:
            env.step(action)
        env.render()
        action = agent.act(env.state, env.reward)
        returns += env.reward

def load_and_watch(dir, env):
    watch(GreedyAgent.load(dir, env), env)

class GreedyAgent(Agent):
    def __init__(
            self,
            action_space,
            feature=None,
            q=None,
            policy=None
    ):
        self.action_space = action_space
        self.feature = feature
        self.policy = None
        if policy:
            self.policy = policy
        else:
            self.policy = q
        if not self.policy:
            raise TypeError('GreedyAgent must have either policy or q function')

    def act(self, state, reward):
        if self.feature:
            state = self.feature(state)
        if isinstance(self.action_space, gym.spaces.Discrete):
            return torch.argmax(self.policy(state), dim=1)
        if isinstance(self.action_space, gym.spaces.Box):
            return (
                self.policy(state)
                .cpu()
                .detach()
                .numpy()
                .reshape(-1)[0:len(self.action_space.shape[0])]
            )
        raise TypeError('Unknown action space')

    @staticmethod
    def load(dirname, env):
        feature = None
        policy = None
        q = None
        for filename in os.listdir(dirname):
            if filename == 'feature.pt':
                feature = torch.load(os.path.join(dirname, filename)).to(env.device)
            if filename == 'policy.pt':
                policy = torch.load(os.path.join(dirname, filename)).to(env.device)
            if filename == 'q.pt':
                q = torch.load(os.path.join(dirname, filename)).to(env.device)
        return GreedyAgent(
            env.action_space,
            feature=feature,
            policy=policy,
            q=q,
        )
