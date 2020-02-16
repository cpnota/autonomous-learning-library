import numpy as np
import torch
from torch.nn.functional import mse_loss
from ._agent import Agent


class DQN(Agent):
    '''
    Deep Q-Network (DQN).
    DQN was one of the original deep reinforcement learning algorithms.
    It extends the ideas behind Q-learning to work well with modern convolution networks.
    The core innovation is the use of a replay buffer, which allows the use of batch-style
    updates with decorrelated samples. It also uses a "target" network in order to
    improve the stability of updates.
    https://www.nature.com/articles/nature14236

    Args:
        q (QNetwork): An Approximation of the Q function.
        policy (GreedyPolicy): A policy derived from the Q-function.
        replay_buffer (ReplayBuffer): The experience replay buffer.
        discount_factor (float): Discount factor for future rewards.
        exploration (float): The probability of choosing a random action.
        loss (function): The weighted loss function to use.
        minibatch_size (int): The number of experiences to sample in each training update.
        n_actions (int): The number of available actions.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        update_frequency (int): Number of timesteps per training update.
    '''
    def __init__(self,
                 q,
                 replay_buffer,
                 discount_factor=0.99,
                 exploration=0.1,
                 loss=mse_loss,
                 minibatch_size=32,
                 n_actions=None,
                 replay_start_size=5000,
                 update_frequency=1,
                 ):
        # objects
        self.q = q
        self.replay_buffer = replay_buffer
        self.loss = staticmethod(loss)
        # hyperparameters
        self.discount_factor = discount_factor
        self.exploration = exploration
        self.minibatch_size = minibatch_size
        self.n_actions = n_actions
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        # private
        self._state = None
        self._action = None
        self._frames_seen = 0

    def act(self, state, reward):
        self.replay_buffer.store(self._state, self._action, reward, state)
        self._train()
        self._state = state
        self._action = self._choose_action(state)
        return self._action

    def eval(self, state, _):
        return self._greedy_action(state)

    def _choose_action(self, state):
        if self._should_explore():
            return torch.randint(self.n_actions, (len(state),), device=self.q.device)
        return self._greedy_action(state)

    def _should_explore(self):
        return (
            len(self.replay_buffer) < self.replay_start_size
            or np.random.rand() < self.exploration
        )

    def _greedy_action(self, state):
        return torch.argmax(self.q.eval(state), dim=1)

    def _train(self):
        if self._should_train():
            # sample transitions from buffer
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(self.minibatch_size)
            # forward pass
            values = self.q(states, actions)
            # compute targets
            targets = rewards + self.discount_factor * torch.max(self.q.target(next_states), dim=1)[0]
            # compute loss
            loss = self.loss(values, targets)
            # backward pass
            self.q.reinforce(loss)

    def _should_train(self):
        self._frames_seen += 1
        return (self._frames_seen > self.replay_start_size and
                self._frames_seen % self.update_frequency == 0)
