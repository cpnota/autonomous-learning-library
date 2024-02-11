import torch
from torch.nn.functional import mse_loss

from all.logging import DummyLogger
from all.memory import GeneralizedAdvantageBuffer

from ._parallel_agent import ParallelAgent
from .a2c import A2CTestAgent


class PPO(ParallelAgent):
    """
    Proximal Policy Optimization (PPO).
    PPO is an actor-critic style policy gradient algorithm that allows for the reuse of samples
    by using importance weighting. This often increases sample efficiency compared to algorithms
    such as A2C. To avoid overfitting, PPO uses a special "clipped" objective that prevents
    the algorithm from changing the current policy too quickly.

    Args:
        features (FeatureNetwork): Shared feature layers.
        v (VNetwork): Value head which approximates the state-value function.
        policy (StochasticPolicy): Policy head which outputs an action distribution.
        discount_factor (float): Discount factor for future rewards.
        entropy_loss_scaling (float): Contribution of the entropy loss to the total policy loss.
        epochs (int): Number of times to reuse each sample.
        lam (float): The Generalized Advantage Estimate (GAE) decay parameter.
        minibatches (int): The number of minibatches to split each batch into.
        compute_batch_size (int): The batch size to use for computations that do not need backpropogation.
        n_envs (int): Number of parallel actors/environments.
        n_steps (int): Number of timesteps per rollout. Updates are performed once per rollout.
        logger (Logger): Used for logging.
    """

    def __init__(
        self,
        features,
        v,
        policy,
        discount_factor=0.99,
        entropy_loss_scaling=0.01,
        epochs=4,
        epsilon=0.2,
        lam=0.95,
        minibatches=4,
        compute_batch_size=256,
        n_envs=None,
        n_steps=4,
        logger=DummyLogger(),
    ):
        if n_envs is None:
            raise RuntimeError("Must specify n_envs.")
        # objects
        self.features = features
        self.v = v
        self.policy = policy
        self.logger = logger
        # hyperparameters
        self.discount_factor = discount_factor
        self.entropy_loss_scaling = entropy_loss_scaling
        self.epochs = epochs
        self.epsilon = epsilon
        self.lam = lam
        self.minibatches = minibatches
        self.compute_batch_size = compute_batch_size
        self.n_envs = n_envs
        self.n_steps = n_steps
        # private
        self._states = None
        self._actions = None
        self._batch_size = n_envs * n_steps
        self._buffer = self._make_buffer()

    def act(self, states):
        self._buffer.store(self._states, self._actions, states.reward)
        self._train(states)
        self._states = states
        self._actions = self.policy.no_grad(self.features.no_grad(states)).sample()
        return self._actions

    def eval(self, states):
        return self.policy.eval(self.features.eval(states))

    def _train(self, next_states):
        if len(self._buffer) >= self._batch_size:
            # load trajectories from buffer
            states, actions, advantages = self._buffer.advantages(next_states)

            # compute target values
            features = states.batch_execute(
                self.compute_batch_size, self.features.no_grad
            )
            features["actions"] = actions
            pi_0 = features.batch_execute(
                self.compute_batch_size,
                lambda s: self.policy.no_grad(s).log_prob(s["actions"]),
            )
            targets = (
                features.batch_execute(self.compute_batch_size, self.v.no_grad)
                + advantages
            )

            # train for several epochs
            for _ in range(self.epochs):
                self._train_epoch(states, actions, advantages, targets, pi_0)

    def _train_epoch(self, states, actions, advantages, targets, pi_0):
        # randomly permute the indexes to generate minibatches
        minibatch_size = int(self._batch_size / self.minibatches)
        indexes = torch.randperm(self._batch_size)
        for n in range(self.minibatches):
            # load the indexes for the minibatch
            first = n * minibatch_size
            last = first + minibatch_size
            i = indexes[first:last]

            # perform a single training step
            self._train_minibatch(
                states[i], actions[i], pi_0[i], advantages[i], targets[i]
            )

    def _train_minibatch(self, states, actions, pi_0, advantages, targets):
        # forward pass
        features = self.features(states)
        values = self.v(features)
        distribution = self.policy(features)
        pi_i = distribution.log_prob(actions)

        # compute losses
        value_loss = mse_loss(values, targets)
        policy_gradient_loss = self._clipped_policy_gradient_loss(
            pi_0, pi_i, advantages
        )
        entropy_loss = -distribution.entropy().mean()
        policy_loss = policy_gradient_loss + self.entropy_loss_scaling * entropy_loss
        loss = value_loss + policy_loss

        # backward pass
        loss.backward()
        self.v.step(loss=value_loss)
        self.policy.step(loss=policy_loss)
        self.features.step()

        # debugging
        self.logger.add_info("entropy", -entropy_loss)
        self.logger.add_info("normalized_value_error", value_loss / targets.var())

    def _clipped_policy_gradient_loss(self, pi_0, pi_i, advantages):
        ratios = torch.exp(pi_i - pi_0)
        surr1 = ratios * advantages
        epsilon = self.epsilon
        surr2 = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon) * advantages
        return -torch.min(surr1, surr2).mean()

    def _make_buffer(self):
        return GeneralizedAdvantageBuffer(
            self.v,
            self.features,
            self.n_steps,
            self.n_envs,
            discount_factor=self.discount_factor,
            lam=self.lam,
            compute_batch_size=self.compute_batch_size,
        )


PPOTestAgent = A2CTestAgent
