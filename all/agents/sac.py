import torch
from all.experiments import DummyWriter
from ._agent import Agent

class SAC(Agent):
    def __init__(self,
                 policy,
                 q_1,
                 q_2,
                 v,
                 replay_buffer,
                 entropy_regularizer=0.01,
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
        self.entropy_regularizer = entropy_regularizer
        # data
        self.env = None
        self.state = None
        self.action = None
        self.frames_seen = 0

    def act(self, state, reward):
        self._store_transition(state, reward)
        self._train()
        self.state = state
        with torch.no_grad():
            self.action = self.policy(state)
        return self.action

    def _store_transition(self, state, reward):
        if self.state and not self.state.done:
            self.frames_seen += 1
            self.replay_buffer.store(self.state, self.action, reward, state)

    def _train(self):
        if self._should_train():
            # randomly sample a batch of transitions
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(
                self.minibatch_size)
            actions = torch.cat(actions)

            # compute targets for Q and V
            with torch.no_grad():
                _actions, _log_probs = self.policy(states, log_prob=True)
                q_targets = rewards + self.discount_factor * self.v.eval(next_states)
                v_targets = torch.min(
                    self.q_1.eval(states, _actions),
                    self.q_2.eval(states, _actions),
                ) - self.entropy_regularizer * _log_probs
                self.writer.add_loss('entropy', -_log_probs.mean())
                self.writer.add_loss('v_mean', v_targets.mean())
                self.writer.add_loss('r_mean', rewards.mean())

            # update Q-functions
            q_1_errors = q_targets - self.q_1(states, actions)
            self.q_1.reinforce(q_1_errors)
            q_2_errors = q_targets - self.q_2(states, actions)
            self.q_2.reinforce(q_2_errors)

            # update V-function
            v_errors = v_targets - self.v(states)
            self.v.reinforce(v_errors)

            # train policy
            _actions, _log_probs = self.policy(states, log_prob=True)

            loss = -(
                self.q_1(states, _actions, detach=False)
                - self.entropy_regularizer * _log_probs
            ).mean()
            loss.backward()
            self.policy.step()
            self.q_1.zero_grad()

    def _should_train(self):
        return (self.frames_seen > self.replay_start_size and
                self.frames_seen % self.update_frequency == 0)
