from all.approximation.action import DiscreteLinearApproximation
from all.approximation.bases import FourierBasis
from all.policies import Greedy
from all.agents import Sarsa as SarsaAgent

def Sarsa(env, alpha=0.05, epsilon=0.1, order=1):
  num_actions = env.env.action_space.n
  basis = FourierBasis(env.env.observation_space, order)
  action_approximation = DiscreteLinearApproximation(alpha, basis, num_actions)
  policy = Greedy(action_approximation, epsilon=epsilon)
  return SarsaAgent(action_approximation, policy)
