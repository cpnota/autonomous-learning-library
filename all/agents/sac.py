import torch
from torch.nn.functional import mse_loss

from all.logging import DummyLogger

from ._agent import Agent


class SAC(Agent):
    """
    Soft Actor-Critic (SAC).
    SAC is a proposed improvement to DDPG that replaces the standard
    mean-squared Bellman error (MSBE) objective with a "maximum entropy"
    objective that impoves exploration. It also uses a few other tricks,
    such as the "Clipped Double-Q Learning" trick introduced by TD3.
    This implementation uses automatic temperature adjustment to replace the
    difficult to set temperature parameter with a more easily tuned
    entropy target parameter.
    https://arxiv.org/abs/1801.01290

    Args:
        policy (DeterministicPolicy): An Approximation of a deterministic policy.
        q1 (QContinuous): An Approximation of the continuous action Q-function.
        q2 (QContinuous): An Approximation of the continuous action Q-function.
        replay_buffer (ReplayBuffer): The experience replay buffer.
        discount_factor (float): Discount factor for future rewards.
        entropy_target (float): The desired entropy of the policy. Usually -env.action_space.shape[0]
        minibatch_size (int): The number of experiences to sample in each training update.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        temperature_initial (float): The initial temperature used in the maximum entropy objective.
        update_frequency (int): Number of timesteps per training update.
    """

    def __init__(
        self,
        policy,
        q1,
        q2,
        replay_buffer,
        discount_factor=0.99,
        entropy_backups=True,
        entropy_target=-2.0,
        lr_temperature=1e-4,
        minibatch_size=32,
        replay_start_size=5000,
        temperature_initial=0.1,
        update_frequency=1,
        logger=DummyLogger(),
    ):
        # objects
        self.policy = policy
        self.q1 = q1
        self.q2 = q2
        self.replay_buffer = replay_buffer
        self.logger = logger
        # hyperparameters
        self.discount_factor = discount_factor
        self.entropy_backups = entropy_backups
        self.entropy_target = entropy_target
        self.lr_temperature = lr_temperature
        self.minibatch_size = minibatch_size
        self.replay_start_size = replay_start_size
        self.temperature = temperature_initial
        self.update_frequency = update_frequency
        # private
        self._state = None
        self._action = None
        self._frames_seen = 0

    def act(self, state):
        self.replay_buffer.store(self._state, self._action, state)
        self._train()
        self._state = state
        self._action = self.policy.no_grad(state)[0]
        return self._action

    def _train(self):
        if self._should_train():
            # sample from replay buffer
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(
                self.minibatch_size
            )
            discount_factor = self.discount_factor

            # compute targets for Q and V
            next_actions, next_log_probs = self.policy.no_grad(next_states)
            q_targets = rewards + discount_factor * torch.min(
                self.q1.target(next_states, next_actions),
                self.q2.target(next_states, next_actions),
            )
            if self.entropy_backups:
                q_targets -= discount_factor * self.temperature * next_log_probs

            # update Q and V-functions
            q1_loss = mse_loss(self.q1(states, actions), q_targets)
            self.q1.reinforce(q1_loss)
            q2_loss = mse_loss(self.q2(states, actions), q_targets)
            self.q2.reinforce(q2_loss)

            # update policy
            new_actions, new_log_probs = self.policy(states)
            q_values = self.q1(states, new_actions)
            loss = -(q_values - self.temperature * new_log_probs).mean()
            self.policy.reinforce(loss)
            self.q1.zero_grad()

            # adjust temperature
            temperature_grad = (
                new_log_probs + self.entropy_target
            ).mean() * self.temperature
            self.temperature = max(
                0, self.temperature + self.lr_temperature * temperature_grad.detach()
            )

            # additional debugging info
            self.logger.add_info("entropy", -new_log_probs.mean())
            self.logger.add_info("q_values", q_values.mean())
            self.logger.add_info("rewards", rewards.mean())
            self.logger.add_info("normalized_q1_error", q1_loss / q_targets.var())
            self.logger.add_info("normalized_q2_error", q2_loss / q_targets.var())
            self.logger.add_info("temperature", self.temperature)
            self.logger.add_info("temperature_grad", temperature_grad)

    def _should_train(self):
        self._frames_seen += 1
        return (
            self._frames_seen > self.replay_start_size
            and self._frames_seen % self.update_frequency == 0
        )


class SACTestAgent(Agent):
    def __init__(self, policy):
        self.policy = policy

    def act(self, state):
        action, log_prob = self.policy.eval(state)
        return action
