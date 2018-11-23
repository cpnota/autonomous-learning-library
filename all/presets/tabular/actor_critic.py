from all.approximation.value.state import TabularStateValue
from all.policies import SoftmaxTabular
from all.agents import ActorCritic


def actor_critic(env, alpha=0.2):
    v = TabularStateValue(alpha, env.state_space)
    policy = SoftmaxTabular(alpha, env.state_space, env.action_space)
    return ActorCritic(v, policy)
