from timeit import default_timer as timer

import numpy as np

from all.logging import ExperimentLogger

from .experiment import Experiment


class SingleEnvExperiment(Experiment):
    """An Experiment object for training and testing agents that interact with one environment at a time."""

    def __init__(
        self,
        preset,
        env,
        name=None,
        train_steps=float("inf"),
        logdir="runs",
        quiet=False,
        render=False,
        save_freq=100,
        verbose=True,
    ):
        self._name = name if name is not None else preset.name
        super().__init__(
            self._make_logger(logdir, self._name, env.name, verbose), quiet
        )
        self._logdir = logdir
        self._preset = preset
        self._agent = self._preset.agent(logger=self._logger, train_steps=train_steps)
        self._env = env
        self._render = render
        self._frame = 1
        self._episode = 1
        self._save_freq = save_freq

        if render:
            self._env.render(mode="human")

    @property
    def frame(self):
        return self._frame

    @property
    def episode(self):
        return self._episode

    def train(self, frames=np.inf, episodes=np.inf):
        while not self._done(frames, episodes):
            self._run_training_episode()
        if len(self._returns100) > 0:
            self._logger.add_summary("returns100", self._returns100)

    def test(self, episodes=100):
        test_agent = self._preset.test_agent()
        returns = []
        episode_lengths = []
        for episode in range(episodes):
            episode_return, episode_length = self._run_test_episode(test_agent)
            returns.append(episode_return)
            episode_lengths.append(episode_length)
            self._log_test_episode(episode, episode_return, episode_length)
        self._log_test(returns, episode_lengths)
        return returns

    def _run_training_episode(self):
        # initialize timer
        start_time = timer()
        start_frame = self._frame

        # initialize the episode
        state = self._env.reset()
        action = self._agent.act(state)
        returns = 0
        episode_length = 0

        # loop until the episode is finished
        while not state.done:
            if self._render:
                self._env.render()
            state = self._env.step(action)
            action = self._agent.act(state)
            returns += state.reward
            episode_length += 1
            self._frame += 1

        # stop the timer
        end_time = timer()
        fps = (self._frame - start_frame) / (end_time - start_time)

        # log the results
        self._log_training_episode(returns, episode_length, fps)
        self._save_model()

        # update experiment state
        self._episode += 1

    def _run_test_episode(self, test_agent):
        # initialize the episode
        state = self._env.reset()
        action = test_agent.act(state)
        returns = 0
        episode_length = 0

        # loop until the episode is finished
        while not state.done:
            if self._render:
                self._env.render()
            state = self._env.step(action)
            action = test_agent.act(state)
            returns += state.reward
            episode_length += 1

        return returns, episode_length

    def _done(self, frames, episodes):
        return self._frame > frames or self._episode > episodes

    def _make_logger(self, logdir, agent_name, env_name, verbose):
        return ExperimentLogger(
            self, agent_name, env_name, verbose=verbose, logdir=logdir
        )

    def _save_model(self):
        if self._save_freq != float("inf") and self._episode % self._save_freq == 0:
            self.save()
