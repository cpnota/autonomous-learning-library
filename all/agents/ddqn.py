import torch
from all.nn import weighted_mse_loss
from ._agent import Agent
from .dqn import DQNTestAgent


class DDQN(Agent):
    '''
    Double Deep Q-Network (DDQN).
    DDQN is an enchancment to DQN that uses a "double Q-style" update,
    wherein the online network is used to select target actions
    and the target network is used to evaluate these actions.
    https://arxiv.org/abs/1509.06461
    This agent also adds support for weighted replay buffers, such
    as priotized experience replay (PER).
    https://arxiv.org/abs/1511.05952

    Args:
        q (QNetwork): An Approximation of the Q function.
        policy (GreedyPolicy): A policy derived from the Q-function.
        replay_buffer (ReplayBuffer): The experience replay buffer.
        discount_factor (float): Discount factor for future rewards.
        loss (function): The weighted loss function to use.
        minibatch_size (int): The number of experiences to sample in each training update.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        update_frequency (int): Number of timesteps per training update.
    '''

    def __init__(self,
                 q,
                 policy,
                 replay_buffer,
                 discount_factor=0.99,
                 loss=weighted_mse_loss,
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
        self._action = self.policy.no_grad(state)
        return self._action

    def eval(self, state):
        return self.policy.eval(state)

    def _train(self):
        if self._should_train():
            # sample transitions from buffer
            (states, actions, rewards, next_states, weights) = self.replay_buffer.sample(self.minibatch_size)
            # forward pass
            values = self.q(states, actions)
            # compute targets
            next_actions = torch.argmax(self.q.no_grad(next_states), dim=1)
            targets = rewards + self.discount_factor * self.q.target(next_states, next_actions)
            # compute loss
            loss = self.loss(values, targets, weights)
            # backward pass
            self.q.reinforce(loss)
            # update replay buffer priorities
            td_errors = targets - values
            self.replay_buffer.update_priorities(td_errors.abs())

    def _should_train(self):
        self._frames_seen += 1
        return self._frames_seen > self.replay_start_size and self._frames_seen % self.update_frequency == 0


DDQNTestAgent = DQNTestAgent
