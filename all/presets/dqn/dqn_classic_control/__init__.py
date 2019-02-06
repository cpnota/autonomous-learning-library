# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch.optim import Adam
from all.approximation import TabularActionValue
from all.agents import DQN
from all.policies import GreedyPolicy
from ..model import deep_q_classic_control

def dqn_cc(env):
    model = deep_q_classic_control(env)
    optimizer = Adam(model.parameters(), lr=1e-4)
    q = TabularActionValue(model, optimizer)
    policy = GreedyPolicy(q, annealing_time=10000)
    return DQN(q, policy, replay_buffer_size=10000, prefetch=1000)

__all__ = ["dqn_cc"]
