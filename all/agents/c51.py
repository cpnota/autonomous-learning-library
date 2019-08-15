import torch
import numpy as np
from all.logging import DummyWriter
from ._agent import Agent


class C51(Agent):
    """
    Double Deep Q-Network

    In additional to the introduction of "double" Q-learning,
    this agent also supports prioritized experience replay
    if replay_buffer is a prioritized buffer.
    """

    def __init__(
            self,
            q_dist,
            replay_buffer,
            exploration=0.02,
            discount_factor=0.99,
            minibatch_size=32,
            replay_start_size=5000,
            update_frequency=1,
            writer=DummyWriter()
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
        self.writer = writer

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
            return torch.randint(
                self.q_dist.n_actions, (len(state),), device=self.q_dist.device
            )
        return self._best_actions(state).view((1))

    def _train(self):
        if self._should_train():
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(
                self.minibatch_size
            )
            # choose best action from online network, double-q style
            next_actions = self._best_actions(next_states)
            # compute the target distribution
            next_dist = self.q_dist.target(next_states, next_actions)
            target_dist = self._project_target_distribution(rewards, next_dist)
            # apply update
            probs = self.q_dist(states, actions)
            self.writer.add_loss('q/mean', (probs * self.q_dist.atoms).sum(dim=1).mean())
            self.q_dist.reinforce(target_dist)

    def _best_actions(self, states):
        probs = self.q_dist.eval(states)
        q_values = (probs * self.q_dist.atoms).sum(dim=2)
        return torch.argmax(q_values, dim=1)

    def _project_target_distribution(self, rewards, dist):
        # pylint: disable=invalid-name
        target_dist = dist * 0
        atoms = self.q_dist.atoms
        v_min = atoms[0]
        v_max = atoms[-1]
        delta_z = atoms[1] - atoms[0]
        tz_j = (rewards.view((-1, 1)) + self.discount_factor * atoms).clamp(v_min, v_max)
        bj = (tz_j - v_min) / delta_z
        l = bj.floor()
        u = bj.ceil()
        target_dist[:, l.long()] += dist * (u - bj)
        target_dist[:, u.long()] += dist * (bj - l)
        return target_dist

    def _should_train(self):
        return (
            self.frames_seen > self.replay_start_size
            and self.frames_seen % self.update_frequency == 0
        )
