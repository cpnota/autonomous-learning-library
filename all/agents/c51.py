import torch
import numpy as np
from all.logging import DummyWriter
from ._agent import Agent

torch.set_printoptions(threshold=10000)

class C51(Agent):
    """
    Implementation of C51, a categorical DQN agent

    The 51 refers to the number of atoms used in the
    categorical distribution used to estimate the
    value distribution. Thought this is the canonical
    name of the agent, this agent is compatible with
    any number of atoms.

    Also note that this implementation uses a "double q"
    style update, which is believed to be less prone
    towards overestimation.
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
        return self._best_actions(state)

    def _train(self):
        if self._should_train():
            (states, actions, rewards, next_states, weights) = self.replay_buffer.sample(
                self.minibatch_size
            )
            actions = torch.cat(actions)
            # choose best action from online network, double-q style
            next_actions = self._best_actions(next_states)
            # compute the distribution at the next state
            next_dist = self.q_dist.target(next_states, next_actions)
            # shift the atoms in the next distribution
            shifted_atoms = (rewards.view((-1, 1)) + self.discount_factor * self.q_dist.atoms)
            # project the disribution back on the original set of atoms
            target_dist = self.q_dist.project(next_dist, shifted_atoms)
            # apply update
            dist = self.q_dist(states, actions, detach=False)
            loss = self._loss(dist, target_dist, weights)
            loss.backward()
            self.q_dist.step()
            # useful for debugging
            self.writer.add_loss('q_dist', loss.detach())
            self.writer.add_loss('q_mean', (dist.detach() * self.q_dist.atoms).sum(dim=1).mean())

    def _best_actions(self, states):
        probs = self.q_dist.eval(states)
        q_values = (probs * self.q_dist.atoms).sum(dim=2)
        return torch.argmax(q_values, dim=1)

    def _should_train(self):
        return (
            self.frames_seen > self.replay_start_size
            and self.frames_seen % self.update_frequency == 0
        )

    def _loss(self, dist, target_dist, weights):
        log_dist = torch.log(torch.clamp(dist, min=1e-5))
        loss_v = log_dist * target_dist
        losses = -loss_v.sum(dim=-1)
        # before aggregating, update priorities
        self.replay_buffer.update_priorities(losses.detach())
        # aggregate
        return (weights * losses).mean()
