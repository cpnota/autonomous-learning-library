# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch import nn
from torch.optim import Adam
from all.layers import Flatten
from all.agents import Sarsa
from all.approximation import QNetwork
from all.policies import GreedyPolicy

def fc_net(env, frames=1):
    return nn.Sequential(
        Flatten(),
        nn.Linear(env.state_space.shape[0] * frames, 256),
        nn.Tanh(),
        nn.Linear(256, env.action_space.n)
    )

def sarsa(
        lr=1e-3,
        epsilon=0.1
):
    def _sarsa(env):
        model = fc_net(env)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QNetwork(model, optimizer, env.action_space.n)
        policy = GreedyPolicy(q, env.action_space.n, annealing_time=1, final_epsilon=epsilon)
        return Sarsa(q, policy)
    return _sarsa

__all__ = ["sarsa"]
