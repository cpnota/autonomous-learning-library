# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from all.approximation import TabularActionValue
from all.agents import Sarsa
from all.policies import GreedyPolicy
from ..model import deep_q_atari


def sarsa(env):
    model = deep_q_atari(env)
    q = TabularActionValue(model)
    policy = GreedyPolicy(q, epsilon=0.1)
    return Sarsa(q, policy)


__all__ = ["sarsa"]
