# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch import nn
from torch.optim import Adam
from all.layers import Flatten
from all.approximation import QTabular
from all.agents import DQN
from all.policies import GreedyPolicy

def conv_net(env, frames=4):
    return nn.Sequential(
        nn.Conv2d(frames, 16, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(2592, 256),
        nn.ReLU(),
        nn.Linear(256, env.action_space.n)
    )

def dqn(env):
    model = conv_net(env)
    optimizer = Adam(model.parameters(), lr=1e-4)
    q = QTabular(model, optimizer, target_update_frequency=250)
    policy = GreedyPolicy(q, annealing_time=250000, initial_epsilon=1.00, final_epsilon=0.02)
    return DQN(q, policy)

__all__ = ["dqn"]
