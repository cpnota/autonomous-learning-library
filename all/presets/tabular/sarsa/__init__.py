# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from all.approximation import TabularActionValue
from all.agents import Sarsa
from all.policies import GreedyPolicy
from ..models import QTable

def sarsa(env):
    model = QTable(env.state_space.n, env.action_space.n)
    q = TabularActionValue(model)
    policy = GreedyPolicy(q, epsilon=0.1)
    return Sarsa(q, policy)

__all__ = ["sarsa"]
