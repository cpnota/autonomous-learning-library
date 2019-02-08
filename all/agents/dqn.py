import torch
from ..utils import ReplayBuffer
from .abstract import Agent


class DQN(Agent):
    def __init__(self,
                 q,
                 policy,
                 replay_buffer_size=100000,
                 minibatch_size=4 * 32,
                 gamma=0.99,
                 prefetch_size=10000,
                 update_frequency=4
                 ):
        # objects
        self.q = q
        self.policy = policy
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        # hyperparameters
        self.prefetch_size = prefetch_size
        self.update_frequency = update_frequency
        self.minibatch_size = minibatch_size
        self.gamma = gamma
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
        next_state = self.env.state
        self.replay_buffer.store(self.state, self.action, next_state, self.env.reward)

    def should_train(self):
        return (self.frames_seen > self.prefetch_size
                and self.frames_seen % self.update_frequency == 0)

    def train(self):
        (states, actions, next_states, rewards) = self.replay_buffer.sample(self.minibatch_size)
        td_errors = (
            rewards
            + self.gamma * torch.max(self.q.eval(next_states), dim=1)[0]
            - self.q(states, actions)
        )
        self.q.reinforce(td_errors)
