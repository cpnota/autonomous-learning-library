import torch
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from ._agent import Agent

class DDPG(Agent):
    def __init__(self,
                 q,
                 policy,
                 replay_buffer,
                 action_space,
                 noise=0.1,
                 discount_factor=0.99,
                 minibatch_size=32,
                 replay_start_size=5000,
                 update_frequency=1,
                 ):
        # objects
        self.q = q
        self.policy = policy
        self.replay_buffer = replay_buffer
        # hyperparameters
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        # private
        self._noise = Normal(0, noise * torch.tensor((action_space.high - action_space.low) / 2).to(policy.device))
        self._low = torch.tensor(action_space.low, device=policy.device)
        self._high = torch.tensor(action_space.high, device=policy.device)
        self._state = None
        self._action = None
        self._frames_seen = 0

    def act(self, state, reward):
        self.replay_buffer.store(self._state, self._action, reward, state)
        self._train()
        self._state = state
        self._action = self._choose_action(state)
        return self._action

    def _choose_action(self, state):
        action = self.policy.eval(state)
        action = action + self._noise.sample()
        action = torch.min(action, self._high)
        action = torch.max(action, self._low)
        return action

    def _train(self):
        if self._should_train():
            # sample transitions from buffer
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(self.minibatch_size)

            # train q-network
            q_values = self.q(states, torch.cat(actions))
            targets = rewards + self.discount_factor * self.q.target(next_states, self.policy.target(next_states))
            loss = mse_loss(q_values, targets)
            self.q.reinforce(loss)

            # train policy
            greedy_actions = self.policy(states)
            loss = -self.q(states, greedy_actions).mean()
            self.policy.reinforce(loss)
            self.q.zero_grad()

    def _should_train(self):
        self._frames_seen += 1
        return self._frames_seen > self.replay_start_size and self._frames_seen % self.update_frequency == 0
