from all.approximation.value.action import TabularActionValue
from all.policies import Greedy
from all.agents import Sarsa as SarsaAgent


def sarsa(env, alpha=0.2, epsilon=0.05):
    action_approximation = TabularActionValue(alpha, env.state_space, env.action_space)
    policy = Greedy(action_approximation, epsilon=epsilon)
    return SarsaAgent(action_approximation, policy)
