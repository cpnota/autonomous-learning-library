import random
import torch
from .abstract import Agent


class DQN(Agent):
    def __init__(self,
                 q,
                 policy,
                 frames=4,
                 replay_buffer_size=100000,
                 minibatch_size=32,
                 gamma=0.99
                 ):
        self.q = q
        self.policy = policy
        self.env = None
        self.state = None
        self.action = None
        self.frames = frames
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.frames_seen = 0

    def new_episode(self, env):
        self.env = env
        print('epsilon: ', self.policy.epsilon)
        print('frames_seen: ', self.frames_seen)

    def act(self):
        self.frames_seen += 1
        self.take_action()
        self.store_transition()
        self.train()

    def take_action(self):
        self.state = self.env.state
        self.action = self.policy(self.state)
        self.env.step(self.action)

    def store_transition(self):
        next_state = self.env.state
        self.replay_buffer.store(
            self.state, self.action, next_state, self.env.reward)

    def train(self):
        (states, actions, next_states, rewards) = self.replay_buffer.sample(self.minibatch_size)
        values = self.q(states, actions)
        targets = rewards + self.gamma * \
            torch.max(self.q.eval(next_states), dim=1)[0]
        td_errors = targets - values
        self.q.reinforce(td_errors)


class ReplayBuffer():
    def __init__(self, size):
        self.data = []
        self.size = size

    def store(self, states, action, next_states, reward):
        self.data.append((states, action, next_states, reward))
        if len(self.data) > self.size:
            self.data = self.data[int(self.size / 10):]

    def sample(self, sample_size):
        minibatch = [random.choice(self.data) for _ in range(0, sample_size)]
        states = [sample[0] for sample in minibatch]
        actions = [sample[1] for sample in minibatch]
        next_states = [sample[2] for sample in minibatch]
        rewards = torch.tensor([sample[3] for sample in minibatch]).float()
        return (states, actions, next_states, rewards)
