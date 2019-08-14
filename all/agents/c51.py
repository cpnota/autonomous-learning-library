import torch
import numpy as np
from ._agent import Agent


class C51(Agent):
    '''
    Double Deep Q-Network

    In additional to the introduction of "double" Q-learning,
    this agent also supports prioritized experience replay
    if replay_buffer is a prioritized buffer.
    '''
    def __init__(self,
                 q_dist,
                 replay_buffer,
                 exploration=0.02,
                 discount_factor=0.99,
                 minibatch_size=32,
                 replay_start_size=5000,
                 update_frequency=1
                 ):
        # objects
        self.q_dist = q_dist
        self.replay_buffer = replay_buffer
        # hyperparameters
        self.exploration = exploration
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
        self.action = self._choose_action(state)
        return self.action

    def _store_transition(self, state, reward):
        if self.state and not self.state.done:
            self.frames_seen += 1
            self.replay_buffer.store(self.state, self.action, reward, state)

    def _choose_action(self, state):
        if np.random.rand() < self.exploration:
            return torch.randint(self.q_dist.actions, (len(state),), device=self.q_dist.device)
        probs = self.q_dist.eval(state).squeeze(0)
        q_values = torch.mv(probs, self.q_dist.atoms)
        return torch.argmax(q_values, dim=1)

    def _train(self):
        '''
        TODO
        '''
        # if self._should_train():
        #     (states, actions, rewards, next_states, weights) = self.replay_buffer.sample(
        #         self.minibatch_size)
        #     next_actions = torch.argmax(self.q.eval(next_states), dim=1)
        #     targets = rewards + self.discount_factor * self.q.target(next_states, next_actions)
        #     td_errors = targets - self.q(states, actions)
        #     self.q.reinforce(weights * td_errors)
        #     self.replay_buffer.update_priorities(td_errors)

    def _should_train(self):
        return (self.frames_seen > self.replay_start_size and
                self.frames_seen % self.update_frequency == 0)
