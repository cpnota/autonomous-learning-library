# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from all.approximation.state_value import ContinuousStateValue
from all.agents import REINFORCE
from all.policies import SoftmaxPolicy
from ..models import tabular_action, continuous_state

def reinforce(env):
    v = ContinuousStateValue(continuous_state(env))
    policy = SoftmaxPolicy(tabular_action(env))
    return REINFORCE(v, policy)
