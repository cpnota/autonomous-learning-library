
import torch
import time
import numpy as np
from all.core import State
from .writer import ExperimentWriter, CometWriter
from .experiment import Experiment
from all.environments import VectorEnvironment
from all.agents import ParallelAgent
import gym


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
        self._name = name if name is not None else preset.name
        super().__init__(self._make_writer(logdir, self._name, env.name, write_loss, writer), quiet)
        self._n_envs = preset.n_envs
        if isinstance(env, VectorEnvironment):
            assert self._n_envs == env.num_envs
            self._env = env
        else:
            self._env = env.duplicate(self._n_envs)
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
        num_envs = int(self._env.num_envs)
        returns = np.zeros(num_envs)
        state_array = self._env.reset()
        start_time = time.time()
        completed_frames = 0
        while not self._done(frames, episodes):
            action = self._agent.act(state_array)
            state_array = self._env.step(action)
            self._frame += num_envs
            episodes_completed = state_array.done.type(torch.IntTensor).sum().item()
            completed_frames += num_envs
            returns += state_array.reward.cpu().detach().numpy()
            if episodes_completed > 0:
                dones = state_array.done.cpu().detach().numpy()
                cur_time = time.time()
                fps = completed_frames / (cur_time - start_time)
                completed_frames = 0
                start_time = cur_time
                for i in range(num_envs):
                    if dones[i]:
                        self._log_training_episode(returns[i], fps)
                        returns[i] = 0
            self._episode += episodes_completed

    def test(self, episodes=100):
        test_agent = self._preset.parallel_test_agent()

        # Note that we need to record the first N episodes that are STARTED,
        # not the first N that are completed, or we introduce bias.
        test_returns = []
        episodes_started = self._n_envs
        should_record = [True] * self._n_envs

        # initialize state
        states = self._env.reset()
        returns = states.reward.clone()

        while len(test_returns) < episodes:
            # step the agent and environments
            actions = test_agent.act(states)
            states = self._env.step(actions)
            returns += states.reward

            # record any episodes that have finished
            for i, done in enumerate(states.done):
                if done:
                    if should_record[i] and len(test_returns) < episodes:
                        episode_return = returns[i].item()
                        test_returns.append(episode_return)
                        self._log_test_episode(len(test_returns), episode_return)
                    returns[i] = 0.
                    episodes_started += 1
                    if episodes_started > episodes:
                        should_record[i] = False

        self._log_test(test_returns)
        return test_returns

    def _done(self, frames, episodes):
        return self._frame > frames or self._episode > episodes

    def _make_writer(self, logdir, agent_name, env_name, write_loss, writer):
        if writer == "comet":
            return CometWriter(self, agent_name, env_name, loss=write_loss, logdir=logdir)
        return ExperimentWriter(self, agent_name, env_name, loss=write_loss, logdir=logdir)
