import torch
import numpy as np
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
        self.frames_seen = 0
        self.env = None
        self.state = None
        self.action = None

    def new_episode(self, env):
        self.env = env

    def act(self):
        self.take_action()
        self.store_transition()
        if self.should_train():
            self.train()

    def take_action(self):
        self.state = self.env.state
        self.action = self.policy(self.state)
        self.env.step(self.action)

    def store_transition(self):
        self.frames_seen += 1
        next_state = self.env.state if not self.env.done else None
        self.replay_buffer.store(self.state, self.action, next_state, np.sign(self.env.reward))

    def should_train(self):
        return (self.frames_seen > self.replay_start_size
                and self.frames_seen % self.update_frequency == 0)

    def train(self):
        (states, actions, next_states, rewards) = self.replay_buffer.sample(self.minibatch_size)
        td_error = (
            rewards
            + self.discount_factor * torch.max(self.q.eval(next_states), dim=1)[0]
            - self.q(states, actions)
        )
        self.q.reinforce(td_error)
