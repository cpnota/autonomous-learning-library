from torch.nn.functional import mse_loss
from ._agent import Agent
from ._parallel_agent import ParallelAgent
from .a2c import A2CTestAgent


class VAC(ParallelAgent):
    '''
    Vanilla Actor-Critic (VAC).
    VAC is an implementation of the actor-critic alogorithm found in the Sutton and Barto (2018) textbook.
    This implementation tweaks the algorithm slightly by using a shared feature layer.
    It is also compatible with the use of parallel environments.
    https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf

    Args:
        features (FeatureNetwork): Shared feature layers.
        v (VNetwork): Value head which approximates the state-value function.
        policy (StochasticPolicy): Policy head which outputs an action distribution.
        discount_factor (float): Discount factor for future rewards.
        n_envs (int): Number of parallel actors/environments
        n_steps (int): Number of timesteps per rollout. Updates are performed once per rollout.
        writer (Writer): Used for logging.
    '''

    def __init__(self, features, v, policy, discount_factor=1):
        self.features = features
        self.v = v
        self.policy = policy
        self.discount_factor = discount_factor
        self._features = None
        self._distribution = None
        self._action = None

    def act(self, state):
        self._train(state, state.reward)
        self._features = self.features(state)
        self._distribution = self.policy(self._features)
        self._action = self._distribution.sample()
        return self._action

    def eval(self, state):
        return self.policy.eval(self.features.eval(state))

    def _train(self, state, reward):
        if self._features:
            # forward pass
            values = self.v(self._features)

            # compute targets
            targets = reward + self.discount_factor * self.v.target(self.features.target(state))
            advantages = targets - values.detach()

            # compute losses
            value_loss = mse_loss(values, targets)
            policy_loss = -(advantages * self._distribution.log_prob(self._action)).mean()

            # backward pass
            self.v.reinforce(value_loss)
            self.policy.reinforce(policy_loss)
            self.features.reinforce()


VACTestAgent = A2CTestAgent
