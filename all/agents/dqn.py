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
                 policy,
                 replay_buffer,
                 discount_factor=0.99,
                 loss=mse_loss,
                 minibatch_size=32,
                 replay_start_size=5000,
                 update_frequency=1,
                 ):
        # objects
        self.q = q
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.loss = loss
        # hyperparameters
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        # private
        self._state = None
        self._action = None
        self._frames_seen = 0

    def act(self, state):
        self.replay_buffer.store(self._state, self._action, state)
        self._train()
        self._state = state
        self._action = self.policy.no_grad(state)
        return self._action

    def eval(self, state):
        return self.policy.eval(state)

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
        return (self._frames_seen > self.replay_start_size and self._frames_seen % self.update_frequency == 0)


class DQNTestAgent(Agent):
    def __init__(self, policy):
        self.policy = policy

    def act(self, state):
        return self.policy.eval(state)
