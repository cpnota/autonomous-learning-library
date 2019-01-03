# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch import nn
from all.approximation.value.action.torch import TabularActionValue 
from all.agents import Sarsa
from all.policies.greedy import Greedy

def sarsa(env):
    model = nn.Sequential(
        nn.Linear(env.state_space.shape[0], env.action_space.n)
    )
    q = TabularActionValue(model)
    policy = Greedy(q, epsilon=0.1)
    return Sarsa(q, policy)
