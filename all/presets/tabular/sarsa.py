from all.approximation.value.action import TabularActionValue
from all.approximation.traces import AccumulatingTraces
from all.policies import Greedy
from all.agents import Sarsa as SarsaAgent


def sarsa(env, alpha=0.1, epsilon=0.1, trace_decay_rate=0.1, order=1):
    action_approximation = AccumulatingTraces(
        TabularActionValue(alpha, env.state_space, env.action_space),
        env,
        trace_decay_rate
    )
    policy = Greedy(action_approximation, epsilon=epsilon)
    return SarsaAgent(action_approximation, policy)
