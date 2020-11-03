
from timeit import default_timer as timer
import torch
import numpy as np
from all.core import State
from .writer import ExperimentWriter
from .experiment import Experiment

class ParallelEnvExperiment(Experiment):
    '''An Experiment object for training and testing agents that use parallel training environments.'''
    def __init__(
            self,
            agent,
            env,
            logdir='runs',
            quiet=False,
            render=False,
            write_loss=True
    ):
        super().__init__(self._make_writer(logdir, agent[0].__name__, env.name, write_loss), quiet)
        make_agent, n_envs = agent
        self._envs = env.duplicate(n_envs)
        self._agent = make_agent(self._envs, self._writer)
        self._n_envs = n_envs
        self._render = render

        # training state
        self._returns = []
        self._frame = 1
        self._episode = 1
        self._episode_start_times = [] * self._n_envs
        self._episode_start_frames = [] * self._n_envs

        # test state
        self._test_episodes = 100
        self._test_episodes_started = self._n_envs
        self._test_returns = []
        self._should_save_returns = [True] * self._n_envs

        if render:
            for _env in self._envs:
                _env.render(mode="human")

    @property
    def frame(self):
        return self._frame

    @property
    def episode(self):
        return self._episode

    def train(self, frames=np.inf, episodes=np.inf):
        self._reset()
        while not (self._frame > frames or self._episode > episodes):
            self._step()

    def test(self, episodes=100):
        self._test_reset(episodes)
        while len(self._test_returns) < episodes:
            self._test_step()
        self._log_test(self._test_returns)
        return self._test_returns

    def _reset(self):
        for env in self._envs:
            env.reset()
        rewards = torch.zeros(
            (self._n_envs),
            dtype=torch.float,
            device=self._envs[0].device
        )
        self._returns = rewards
        self._episode_start_times = [timer()] * self._n_envs
        self._episode_start_frames = [self._frame] * self._n_envs

    def _step(self):
        states = self._aggregate_states()
        actions = self._agent.act(states)
        self._step_envs(actions)

    def _step_envs(self, actions):
        for i, env in enumerate(self._envs):
            state = env.state
            if self._render:
                env.render()

            if state.done:
                self._returns[i] += state.reward
                self._log_training_episode(self._returns[i].item(), self._fps(i))
                env.reset()
                self._returns[i] = 0
                self._episode += 1
                self._episode_start_times[i] = timer()
                self._episode_start_frames[i] = self._frame
            else:
                action = actions[i]
                if action is not None:
                    self._returns[i] += state.reward
                    env.step(action)
                    self._frame += 1

    def _test_reset(self, episodes):
        self._reset()
        self._test_episodes = episodes
        self._test_episodes_started = 0
        self._test_returns = []
        self._should_save_returns = [True] * self._n_envs

    def _test_step(self):
        states = self._aggregate_states()
        actions = self._agent.eval(states)
        self._test_step_envs(actions)

    def _test_step_envs(self, actions):
        for i, env in enumerate(self._envs):
            state = env.state
            if self._render:
                env.render()
            if state.done:
                self._returns[i] += state.reward
                if self._should_save_returns[i]:
                    self._test_returns.append(self._returns[i].item())
                    self._log_test_episode(len(self._test_returns), self._returns[i].item())
                if self._test_episodes_started > self._test_episodes:
                    self._should_save_returns[i] = False
                env.reset()
                self._returns[i] = 0
                self._test_episodes_started += 1
            else:
                action = actions[i]
                if action is not None:
                    self._returns[i] += state.reward
                    env.step(action)

    def _aggregate_states(self):
        return State.array([env.state for env in self._envs])

    def _aggregate_rewards(self):
        return torch.tensor(
            [env.state.reward for env in self._envs],
            dtype=torch.float,
            device=self._envs[0].device
        )

    def _fps(self, i):
        end_time = timer()
        return (self._frame - self._episode_start_frames[i]) / (end_time - self._episode_start_times[i])

    def _make_writer(self, logdir, agent_name, env_name, write_loss):
        return ExperimentWriter(self, agent_name, env_name, loss=write_loss, logdir=logdir)
