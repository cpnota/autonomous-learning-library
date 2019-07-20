import torch
from ._agent import Agent
from all.experiments import DummyWriter

class SAC(Agent):
    def __init__(self,
                 policy,
                 q1,
                 q2,
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
        self.q1 = q1
        self.q2 = q2
        self.replay_buffer = replay_buffer
        self.writer=writer
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
        self.action = self.policy(state)
        return self.action

    def _store_transition(self, state, reward):
        if self.state and not self.state.done:
            self.frames_seen += 1
            self.replay_buffer.store(self.state, self.action, reward, state)

    def _train(self):
        if self._should_train():
            # Randomly sample a batch of transitions
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(
                self.minibatch_size)
            actions = torch.cat(actions)

            # resample actions
            _actions = self.policy(states).detach()

            # compute targets for Q and V
            entropy = self.policy.log_prob(_actions).detach()
            q_targets = rewards + self.discount_factor * self.v.eval(next_states)
            v_targets = torch.min(
                self.q1.eval(states, _actions),
                self.q2.eval(states, _actions),
            ) - self.entropy_regularizer * entropy
            self.writer.add_loss('entropy', -entropy.mean())
            self.writer.add_loss('v_mean', v_targets.mean())
            self.writer.add_loss('r_mean', rewards.mean())

            # update Q-functions
            q1_errors = q_targets - self.q1(states, actions)
            q2_errors = q_targets - self.q2(states, actions)
            self.q1.reinforce(q1_errors)
            self.q2.reinforce(q2_errors)

            # update V-function
            v_errors = v_targets - self.v(states)
            self.v.reinforce(v_errors)

            # train policy
            __actions = self.policy(states)
            loss = -(
                self.q1(states, __actions, detach=False)
                - self.entropy_regularizer * self.policy.log_prob(actions)
            ).mean()
            loss.backward()
            self.policy.step()
            self.q1.zero_grad()

    def _should_train(self):
        return (self.frames_seen > self.replay_start_size and
                self.frames_seen % self.update_frequency == 0)
