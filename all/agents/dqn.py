import random
import torch
from .abstract import Agent


class DQN(Agent):
    def __init__(self,
                 q,
                 policy,
                 replay_buffer_size=100000,
                 minibatch_size=32,
                 gamma=0.99,
                 prefetch=10000,
                 update_frequency=4
                 ):
        self.q = q
        self.policy = policy
        self.env = None
        self.state = None
        self.action = None
        self.update_frequency = update_frequency
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.frames_seen = 0
        self.prefetch = prefetch

    def new_episode(self, env):
        self.env = env
        print('epsilon: ', self.policy.epsilon)
        print('frames_seen: ', self.frames_seen)

    def act(self):
        self.take_action()
        self.store_transition()
        if (self.should_train()):
            self.train()

    def take_action(self):
        self.state = self.env.state
        self.action = self.policy(self.state)
        self.env.step(self.action)

    def store_transition(self):
        self.frames_seen += 1
        next_state = self.env.state
        self.replay_buffer.store(
            self.state, self.action, next_state, self.env.reward)

    def should_train(self):
       return self.frames_seen > self.prefetch and self.frames_seen % self.update_frequency

    def train(self):
        (states, actions, next_states, rewards) = self.replay_buffer.sample(self.minibatch_size)
        values = self.q(states, actions)
        targets = rewards + self.gamma * \
            torch.max(self.q.eval(next_states), dim=1)[0]
        td_errors = targets - values
        self.q.reinforce(td_errors)

class ReplayBuffer:
    # https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
    def __init__(self, size):
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0

    def store(self, states, action, next_states, reward):
        self._append((states, action, next_states, reward))
        
    def sample(self, sample_size):
        minibatch = [random.choice(self) for _ in range(0, sample_size)]
        states = [sample[0] for sample in minibatch]
        actions = [sample[1] for sample in minibatch]
        next_states = [sample[2] for sample in minibatch]
        rewards = torch.tensor([sample[3] for sample in minibatch]).float()
        return (states, actions, next_states, rewards)

    def _append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
        
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
