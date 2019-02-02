import random
import torch
from .abstract import Agent

MAX_BUFFER = 100000

def stack(frames):
    return torch.cat(frames).unsqueeze(0) if frames is not None else None

class DQN(Agent):
    def __init__(self, q, policy, frames=4):
        self.q = q
        self.policy = policy
        self.env = None
        self.states = None
        self.action = None
        self.buffer = []
        self.frames = frames

    def new_episode(self, env):
        self.env = env
        self.states = [self.env.state.squeeze(0)] * self.frames

    def act(self):
        self.take_action()
        self.store_transition()
        self.train()

    def take_action(self):
        inputs = stack(self.states)
        self.action = self.policy(inputs)
        self.env.step(self.action)

    def store_transition(self):
        next_states = None if self.env.state is None else self.states[1:] + [self.env.state.squeeze(0)]
        self.add_to_buffer(self.states, self.action, next_states, self.env.reward)
        self.states = next_states

    def add_to_buffer(self, states, action, next_states, reward):
        self.buffer.append((states, action, next_states, reward))
        if len(self.buffer) > MAX_BUFFER:
            self.buffer = self.buffer[int(MAX_BUFFER / 10):]

    def train(self):
        (states, actions, next_states, rewards) = self.sample_minibatch()
        values = self.q(states, actions)
        targets = rewards + 0.99 * torch.max(self.q.eval(next_states), dim=1)[0]
        td_errors = targets - values
        self.q.reinforce(td_errors)

    def sample_minibatch(self):
        minibatch = [random.choice(self.buffer) for _ in range(0, 32)]
        states = [stack(sample[0]) for sample in minibatch]
        actions = [sample[1] for sample in minibatch]
        next_states = [stack(sample[2]) for sample in minibatch]
        rewards = torch.tensor([sample[3] for sample in minibatch]).float()
        return (states, actions, next_states, rewards)
