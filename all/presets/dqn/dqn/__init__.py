# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch.optim import Adam
from all.approximation import TabularActionValue
from all.agents import DQN
from all.policies import GreedyPolicy
from ..model import deep_q_atari

def dqn(env):
    model = deep_q_atari(env)
    optimizer = Adam(model.parameters(), lr=1e-4)
    q = TabularActionValue(model, optimizer)
    policy = GreedyPolicy(q, annealing_time=250000)
    return DQN(q, policy)

__all__ = ["dqn"]
