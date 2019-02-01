import random
import torch
from .abstract import Agent

MAX_BUFFER = 100000

class DQN(Agent):
    def __init__(self, q, policy, frames=4):
        self.q = q
        self.policy = policy
        self.env = None
        self.states = None
        self.buffer = []
        self.frames = frames

    def new_episode(self, env):
        self.env = env
        self.states = [self.env.state.squeeze(0)] * self.frames

    def act(self):
        inputs = torch.cat(self.states).unsqueeze(0)
        action = self.policy(inputs)
        self.env.step(action)

        next_states = None if self.env.state is None else self.states[1:] + [self.env.state.squeeze(0)]
        self.record(self.states, action, next_states, self.env.reward)
        self.update()
        self.states = next_states

    def record(self, states, action, next_states, reward):
        self.buffer.append((states, action, next_states, reward))
        if len(self.buffer) > MAX_BUFFER:
            self.buffer = self.buffer[int(MAX_BUFFER / 10):]


    def update(self):
        minibatch = [random.choice(self.buffer) for _ in range(0, 32)]
        
        states = [torch.cat(sample[0]).unsqueeze(0) for sample in minibatch]
        actions = [sample[1] for sample in minibatch]
        next_states = [(torch.cat(sample[2]).unsqueeze(0) if sample[2] is not None else None) for sample in minibatch]
        rewards = torch.tensor([sample[3] for sample in minibatch]).float()

        values = self.q(states, actions)
        targets = rewards + torch.max(self.q.eval(next_states), dim=1)[0]
        td_errors = targets - values

        self.q.reinforce(td_errors)
