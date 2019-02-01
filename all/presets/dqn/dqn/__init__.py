# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from all.approximation import TabularActionValue
from all.agents import DQN
from all.policies import GreedyPolicy
from ..model import deep_q_atari

def dqn(env):
    model = deep_q_atari(env)
    q = TabularActionValue(model)
    policy = GreedyPolicy(q)
    return DQN(q, policy)

__all__ = ["dqn"]
