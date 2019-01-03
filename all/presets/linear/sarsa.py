# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from all.approximation.action_value import TabularActionValue
from all.agents import Sarsa
from all.policies import GreedyPolicy
from .models import tabular_action

def sarsa(env):
    model = tabular_action(env)
    q = TabularActionValue(model)
    policy = GreedyPolicy(q, epsilon=0.1)
    return Sarsa(q, policy)
