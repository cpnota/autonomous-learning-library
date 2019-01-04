# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from all.approximation.state_value import ContinuousStateValue
from all.agents import ActorCritic
from all.policies import SoftmaxPolicy
from ..models import QTable, VTable

def actor_critic(env):
    v = ContinuousStateValue(VTable(env.state_space.n))
    policy = SoftmaxPolicy(QTable(env.state_space.n, env.action_space.n))
    return ActorCritic(v, policy)
