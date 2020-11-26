import torch
from torch.nn.functional import mse_loss
from all.core import State
from ._agent import Agent
from .a2c import A2CTestAgent


class VPG(Agent):
    '''
    Vanilla Policy Gradient (VPG/REINFORCE).
    VPG (also known as REINFORCE) is the least biased implementation of the policy gradient theorem.
    It uses complete episode rollouts as unbiased estimates of the Q-function, rather than the n-step
    rollouts found in most actor-critic algorithms. The state-value function approximation reduces
    varience, but does not introduce any bias. This implementation introduces two tweaks. First,
    it uses a shared feature layer. Second, it introduces the capacity for training on multiple
    episodes at once. These enhancements often improve learning without sacrifice the essential
    character of the algorithm.
    https://link.springer.com/article/10.1007/BF00992696

    Args:
        features (FeatureNetwork): Shared feature layers.
        v (VNetwork): Value head which approximates the state-value function.
        policy (StochasticPolicy): Policy head which outputs an action distribution.
        discount_factor (float): Discount factor for future rewards.
        min_batch_size (int): Updates will occurs when an episode ends after at least
            this many state-action pairs are seen. Set this to a large value in order
            to train on multiple episodes at once.
    '''

    def __init__(
            self,
            features,
            v,
            policy,
            discount_factor=0.99,
            min_batch_size=1
    ):
        self.features = features
        self.v = v
        self.policy = policy
        self.discount_factor = discount_factor
        self.min_batch_size = min_batch_size
        self._current_batch_size = 0
        self._trajectories = []
        self._features = []
        self._log_pis = []
        self._rewards = []

    def act(self, state):
        if not self._features:
            return self._initial(state)
        if not state.done:
            return self._act(state, state.reward)
        return self._terminal(state, state.reward)

    def eval(self, state):
        return self.policy.eval(self.features.eval(state))

    def _initial(self, state):
        features = self.features(state)
        distribution = self.policy(features)
        action = distribution.sample()
        self._features = [features]
        self._log_pis.append(distribution.log_prob(action))
        return action

    def _act(self, state, reward):
        features = self.features(state)
        distribution = self.policy(features)
        action = distribution.sample()
        self._features.append(features)
        self._rewards.append(reward)
        self._log_pis.append(distribution.log_prob(action))
        return action

    def _terminal(self, state, reward):
        self._rewards.append(reward)
        features = State.array(self._features)
        rewards = torch.tensor(self._rewards, device=features.device)
        log_pis = torch.stack(self._log_pis)
        self._trajectories.append((features, rewards, log_pis))
        self._current_batch_size += len(features)
        self._features = []
        self._rewards = []
        self._log_pis = []

        if self._current_batch_size >= self.min_batch_size:
            self._train()

        # have to return something
        return self.policy.no_grad(self.features.no_grad(state)).sample()

    def _train(self):
        # forward pass
        values = torch.cat([
            self.v(features)
            for (features, _, _)
            in self._trajectories
        ])

        # forward passes for log_pis were stored during execution
        log_pis = torch.cat([log_pis for (_, _, log_pis) in self._trajectories])

        # compute targets
        targets = torch.cat([
            self._compute_discounted_returns(rewards)
            for (_, rewards, _)
            in self._trajectories
        ])
        advantages = targets - values.detach()

        # compute losses
        value_loss = mse_loss(values, targets)
        policy_loss = -(advantages * log_pis).mean()

        # backward pass
        self.v.reinforce(value_loss)
        self.policy.reinforce(policy_loss)
        self.features.reinforce()

        # cleanup
        self._trajectories = []
        self._current_batch_size = 0

    def _compute_discounted_returns(self, rewards):
        returns = rewards.clone()
        t = len(returns) - 1
        discounted_return = 0
        for reward in torch.flip(rewards, dims=(0,)):
            discounted_return = reward + self.discount_factor * discounted_return
            returns[t] = discounted_return
            t -= 1
        return returns


VPGTestAgent = A2CTestAgent
