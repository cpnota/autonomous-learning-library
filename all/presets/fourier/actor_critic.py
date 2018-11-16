from all.approximation.value.state import LinearStateValue
from all.approximation.bases import FourierBasis
# from all.approximation.traces import AccumulatingTraces
from all.policies import SoftmaxLinear
from all.agents import ActorCritic


def actor_critic(env, alpha=0.05, order=1):
    num_actions = env.env.action_space.n
    basis = FourierBasis(env.env.observation_space, order)
    v = LinearStateValue(alpha, basis)
    policy = SoftmaxLinear(alpha, basis, num_actions)
    return ActorCritic(v, policy)
