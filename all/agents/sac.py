import torch
from torch.nn.functional import mse_loss
from all.logging import DummyWriter
from ._agent import Agent

class SAC(Agent):
    def __init__(self,
                 policy,
                 q_1,
                 q_2,
                 v,
                 replay_buffer,
                 entropy_target=-2., # usually -action_space.size[0]
                 temperature_initial=0.1,
                 lr_temperature=1e-4,
                 discount_factor=0.99,
                 minibatch_size=32,
                 replay_start_size=5000,
                 update_frequency=1,
                 writer=DummyWriter()
                 ):
        # objects
        self.policy = policy
        self.v = v
        self.q_1 = q_1
        self.q_2 = q_2
        self.replay_buffer = replay_buffer
        self.writer = writer
        # hyperparameters
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        # vars for learning the temperature
        self.entropy_target = entropy_target
        self.temperature = temperature_initial
        self.lr_temperature = lr_temperature
        # data
        self.env = None
        self.state = None
        self.action = None
        self.frames_seen = 0

    def act(self, state, reward):
        self._store_transition(state, reward)
        self._train()
        self.state = state
        self.action = self.policy.eval(state)[0]
        return self.action

    def _store_transition(self, state, reward):
        if self.state and not self.state.done:
            self.frames_seen += 1
            self.replay_buffer.store(self.state, self.action, reward, state)

    def _train(self):
        if self._should_train():
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(self.minibatch_size)
            actions = torch.cat(actions)

            # compute targets for Q and V
            _actions, _log_probs = self.policy.eval(states)
            q_targets = rewards + self.discount_factor * self.v.target(next_states)
            v_targets = torch.min(
                self.q_1.target(states, _actions),
                self.q_2.target(states, _actions),
            ) - self.temperature * _log_probs

            # update Q and V-functions
            self.q_1.reinforce(mse_loss(self.q_1(states, actions), q_targets))
            self.q_2.reinforce(mse_loss(self.q_2(states, actions), q_targets))
            self.v.reinforce(mse_loss(self.v(states), v_targets))

            # update policy
            _actions2, _log_probs2 = self.policy(states)
            loss = (-self.q_1(states, _actions2) + self.temperature * _log_probs2).mean()
            self.policy.reinforce(loss)
            self.q_1.zero_grad()

            # adjust temperature
            temperature_grad = (_log_probs + self.entropy_target).mean()
            self.temperature += self.lr_temperature * temperature_grad.detach()

            # additional debugging info
            self.writer.add_loss('entropy', -_log_probs.mean())
            self.writer.add_loss('v_mean', v_targets.mean())
            self.writer.add_loss('r_mean', rewards.mean())
            self.writer.add_loss('temperature_grad', temperature_grad)
            self.writer.add_loss('temperature', self.temperature)

    def _should_train(self):
        return (self.frames_seen > self.replay_start_size and
                self.frames_seen % self.update_frequency == 0)
