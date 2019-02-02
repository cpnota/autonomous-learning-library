import random
import torch
from .abstract import Agent

def stack(frames):
    return torch.cat(frames, dim=1) if frames is not None else None

class DQN(Agent):
    def __init__(self, q, policy, frames=4, replay_buffer_size=100000):
        self.q = q
        self.policy = policy
        self.env = None
        self.states = None
        self.action = None
        self.frames = frames
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def new_episode(self, env):
        self.env = env
        self.states = [self.env.state] * self.frames

    def act(self):
        self.take_action()
        self.store_transition()
        self.train()

    def take_action(self):
        inputs = stack(self.states)
        self.action = self.policy(inputs)
        self.env.step(self.action)

    def store_transition(self):
        next_states = None if self.env.state is None else self.states[1:] + [self.env.state]
        self.replay_buffer.store(self.states, self.action, next_states, self.env.reward)
        self.states = next_states

    def train(self):
        (states, actions, next_states, rewards) = self.replay_buffer.sample(32)
        values = self.q(states, actions)
        targets = rewards + 0.99 * torch.max(self.q.eval(next_states), dim=1)[0]
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
        states = [stack(sample[0]) for sample in minibatch]
        actions = [sample[1] for sample in minibatch]
        next_states = [stack(sample[2]) for sample in minibatch]
        rewards = torch.tensor([sample[3] for sample in minibatch]).float()
        return (states, actions, next_states, rewards)
