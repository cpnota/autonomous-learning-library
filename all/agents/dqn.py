import torch
from torch.nn.functional import mse_loss
from ._agent import Agent


class DQN(Agent):
    def __init__(self,
                 q,
                 policy,
                 replay_buffer,
                 loss=mse_loss,
                 discount_factor=0.99,
                 minibatch_size=32,
                 replay_start_size=5000,
                 update_frequency=1
                 ):
        # objects
        self.q = q
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.loss = staticmethod(loss)
        # hyperparameters
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        # data
        self.env = None
        self.state = None
        self.action = None
        self.frames_seen = 0

    def act(self, state, reward):
        self._store_transition(state, reward)
        self._train()
        self._state = state
        self._action = self.policy(state)
        return self.action

    def _store_transition(self, state, reward):
        if self.state and not self.state.done:
            self.frames_seen += 1
            self.replay_buffer.store(self.state, self.action, reward, state)

    def _train(self):
        if self._should_train():
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(self.minibatch_size)
            values = self.q(states, actions)
            targets = rewards + self.discount_factor * torch.max(self.q.target(next_states), dim=1)[0]
            loss = self.loss(values, targets)
            self.q.reinforce(loss)

    def _should_train(self):
        return (self.frames_seen > self.replay_start_size and
                self.frames_seen % self.update_frequency == 0)
