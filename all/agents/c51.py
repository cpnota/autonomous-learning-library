import torch
import numpy as np
from all.logging import DummyWriter
from ._agent import Agent


class C51(Agent):
    """
    A categorical DQN agent (C51).
    Rather than making a point estimate of the Q-function,
    C51 estimates a categorical distribution over possible values.
    The 51 refers to the number of atoms used in the
    categorical distribution used to estimate the
    value distribution.
    https://arxiv.org/abs/1707.06887

    Args:
        q_dist (QDist): Approximation of the Q distribution.
        replay_buffer (ReplayBuffer): The experience replay buffer.
        discount_factor (float): Discount factor for future rewards.
        eps (float): Stability parameter for computing the loss function.
        exploration (float): The probability of choosing a random action.
        minibatch_size (int): The number of experiences to sample in each training update.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        update_frequency (int): Number of timesteps per training update.
    """

    def __init__(
            self,
            q_dist,
            replay_buffer,
            discount_factor=0.99,
            eps=1e-5,
            exploration=0.02,
            minibatch_size=32,
            replay_start_size=5000,
            update_frequency=1,
            writer=DummyWriter(),
    ):
        # objects
        self.q_dist = q_dist
        self.replay_buffer = replay_buffer
        self.writer = writer
        # hyperparameters
        self.eps = eps
        self.exploration = exploration
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        # private
        self._state = None
        self._action = None
        self._frames_seen = 0

    def act(self, state):
        self.replay_buffer.store(self._state, self._action, state)
        self._train()
        self._state = state
        self._action = self._choose_action(state)
        return self._action

    def eval(self, state):
        return self._best_actions(self.q_dist.eval(state)).item()

    def _choose_action(self, state):
        if self._should_explore():
            return np.random.randint(0, self.q_dist.n_actions)
        return self._best_actions(self.q_dist.no_grad(state)).item()

    def _should_explore(self):
        return (
            len(self.replay_buffer) < self.replay_start_size
            or np.random.rand() < self.exploration
        )

    def _best_actions(self, probs):
        q_values = (probs * self.q_dist.atoms).sum(dim=-1)
        return torch.argmax(q_values, dim=-1)

    def _train(self):
        if self._should_train():
            # sample transitions from buffer
            states, actions, rewards, next_states, weights = self.replay_buffer.sample(self.minibatch_size)
            # forward pass
            dist = self.q_dist(states, actions)
            # compute target distribution
            target_dist = self._compute_target_dist(next_states, rewards)
            # compute loss
            kl = self._kl(dist, target_dist)
            loss = (weights * kl).mean()
            # backward pass
            self.q_dist.reinforce(loss)
            # update replay buffer priorities
            self.replay_buffer.update_priorities(kl.detach())
            # debugging
            self.writer.add_loss(
                "q_mean", (dist.detach() * self.q_dist.atoms).sum(dim=1).mean()
            )

    def _should_train(self):
        self._frames_seen += 1
        return self._frames_seen > self.replay_start_size and self._frames_seen % self.update_frequency == 0

    def _compute_target_dist(self, states, rewards):
        actions = self._best_actions(self.q_dist.no_grad(states))
        dist = self.q_dist.target(states, actions)
        shifted_atoms = (
            rewards.view((-1, 1)) + self.discount_factor * self.q_dist.atoms
        )
        return self.q_dist.project(dist, shifted_atoms)

    def _kl(self, dist, target_dist):
        log_dist = torch.log(torch.clamp(dist, min=self.eps))
        log_target_dist = torch.log(torch.clamp(target_dist, min=self.eps))
        return (target_dist * (log_target_dist - log_dist)).sum(dim=-1)


class C51TestAgent(Agent):
    def __init__(self, q_dist, n_actions, exploration=0.):
        self.q_dist = q_dist
        self.n_actions = n_actions
        self.exploration = exploration

    def act(self, state):
        if np.random.rand() < self.exploration:
            return np.random.randint(0, self.n_actions)
        q_values = (self.q_dist(state) * self.q_dist.atoms).sum(dim=-1)
        return torch.argmax(q_values, dim=-1)
