from all.approximation.value.action import LinearStateDiscreteActionValue
from all.approximation.bases import FourierBasis
from all.approximation.traces import AccumulatingTraces
from all.policies import Greedy
from all.agents import Sarsa as SarsaAgent


def sarsa(env, alpha=0.05, epsilon=0.1, trace_decay_rate=0.5, order=1):
    num_actions = env.env.action_space.n
    basis = FourierBasis(env.env.observation_space, order)
    action_approximation = AccumulatingTraces(
        LinearStateDiscreteActionValue(alpha, basis, num_actions),
        env,
        trace_decay_rate
    )
    policy = Greedy(action_approximation, epsilon=epsilon)
    return SarsaAgent(action_approximation, policy)
