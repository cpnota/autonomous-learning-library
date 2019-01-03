# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from all.approximation.action_value import TabularActionValue
from all.agents import Sarsa
from all.policies.greedy import Greedy
from .models import tabular_action

def sarsa(env):
    model = tabular_action(env)
    q = TabularActionValue(model)
    policy = Greedy(q, epsilon=0.1)
    return Sarsa(q, policy)
