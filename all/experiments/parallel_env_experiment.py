
from timeit import default_timer as timer
import torch
import numpy as np
from all.core import State
from .writer import ExperimentWriter, CometWriter
from .experiment import Experiment


class ParallelEnvExperiment(Experiment):
    '''An Experiment object for training and testing agents that use parallel training environments.'''

    def __init__(
            self,
            preset,
            env,
            name=None,
            train_steps=float('inf'),
            logdir='runs',
            quiet=False,
            render=False,
            write_loss=True,
            writer="tensorboard"
    ):
        self._name = name if name is not None else preset.__class__.__name__
        super().__init__(self._make_writer(logdir, self._name, env.name, write_loss, writer), quiet)
        self._n_envs = preset.n_envs
        self._envs = env.duplicate(self._n_envs)
        self._preset = preset
        self._agent = preset.agent(writer=self._writer, train_steps=train_steps)
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
        test_agent = self._preset.test_agent()
        env = self._envs[0].duplicate(1)[0]
        returns = []
        for episode in range(episodes):
            episode_return = self._run_test_episode(test_agent, env)
            returns.append(episode_return)
            self._log_test_episode(episode, episode_return)
        self._log_test(returns)
        return returns

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

    def _aggregate_states(self):
        return State.array([env.state for env in self._envs])

    def _aggregate_rewards(self):
        return torch.tensor(
            [env.state.reward for env in self._envs],
            dtype=torch.float,
            device=self._envs[0].device
        )

    def _run_test_episode(self, test_agent, env):
        # initialize the episode
        state = env.reset()
        action = test_agent.act(state)
        returns = 0

        # loop until the episode is finished
        while not state.done:
            if self._render:
                env.render()
            state = env.step(action)
            action = test_agent.act(state)
            returns += state.reward

        return returns

    def _fps(self, i):
        end_time = timer()
        return (self._frame - self._episode_start_frames[i]) / (end_time - self._episode_start_times[i])

    def _make_writer(self, logdir, agent_name, env_name, write_loss, writer):
        if writer == "comet":
            return CometWriter(self, agent_name, env_name, loss=write_loss, logdir=logdir)
        return ExperimentWriter(self, agent_name, env_name, loss=write_loss, logdir=logdir)
