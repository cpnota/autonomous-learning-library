# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from all.approximation.state_value import ContinuousStateValue
from all.agents import ActorCritic
from all.policies import SoftmaxPolicy
from .models import tabular_action, continuous_state

def actor_critic(env):
    v = ContinuousStateValue(continuous_state(env))
    policy = SoftmaxPolicy(tabular_action(env))
    return ActorCritic(v, policy)
