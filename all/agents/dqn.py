import torch
from .abstract import Agent


class DQN(Agent):
    def __init__(self,
                 q,
                 policy,
                 replay_buffer,
                 discount_factor=0.99,
                 minibatch_size=32,
                 replay_start_size=5000,
                 update_frequency=1
                 ):
        # objects
        self.q = q
        self.policy = policy
        self.replay_buffer = replay_buffer
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
        self.state = state
        self.action = self.policy(state)
        return self.action

    def _store_transition(self, state, reward):
        if self.state and not self.state.done:
            self.frames_seen += 1
            self.replay_buffer.store(self.state, self.action, state, reward)

    def _train(self):
        if self._should_train():
            (states, actions, next_states, rewards, weights) = self.replay_buffer.sample(
                self.minibatch_size)
            td_errors = (
                rewards +
                self.discount_factor * torch.max(self.q.eval(next_states), dim=1)[0] -
                self.q(states, actions)
            )
            self.q.reinforce(weights * td_errors)
            self.replay_buffer.update_priorities(td_errors)

    def _should_train(self):
        return (self.frames_seen > self.replay_start_size and
                self.frames_seen % self.update_frequency == 0)
