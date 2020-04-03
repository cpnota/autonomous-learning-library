from abc import ABC, abstractmethod
from collections.abc import Iterable
from timeit import default_timer as timer
import numpy as np
from .writer import ExperimentWriter

class Experiment(ABC):
    def __init__(self, writer, quiet):
        self._writer = writer
        self._quiet = quiet
        self._best_returns = -np.inf
        self._returns100 = []

    @abstractmethod
    def train(self, frames=np.inf, episodes=np.inf):
        pass

    def eval(self, frames=np.inf, episodes=np.inf):
        pass

    @property
    @abstractmethod
    def frame(self):
        pass

    @property
    @abstractmethod
    def episode(self):
        pass

    def _log_training_episode(self, returns, fps):
        if not self._quiet:
            print('episode: {}, frame: {}, fps: {}, returns: {}'.format(self.episode, self.frame, fps, returns))
        if returns > self._best_returns:
            self._best_returns = returns
        self._returns100.append(returns)
        if len(self._returns100) == 100:
            mean = np.mean(self._returns100)
            std = np.std(self._returns100)
            self._writer.add_summary('returns100', mean, std, step="frame")
            self._returns100 = []
        self._writer.add_evaluation('returns/episode', returns, step="episode")
        self._writer.add_evaluation('returns/frame', returns, step="frame")
        self._writer.add_evaluation("returns/max", self._best_returns, step="frame")
        self._writer.add_scalar('fps', fps, step="frame")

    def _log_test_episode(self, episode, returns):
        if not self._quiet:
            print('test episode: {}, returns: {}'.format(episode, returns))


class SingleEnvExperiment(Experiment):
    def __init__(
            self,
            agent,
            env,
            render=False,
            quiet=False,
            write_loss=True
    ):
        super().__init__(self._make_writer(agent.__name__, env.name, write_loss), quiet)
        self._agent = agent(env, self._writer)
        self._env = env
        self._render = render
        self._frame = 1
        self._episode = 1

    def train(self, frames=np.inf, episodes=np.inf):
        while not self._done(frames, episodes):
            self._run_training_episode()

    def test(self, episodes=100):
        returns = []
        for episode in range(episodes):
            episode_return = self._run_test_episode()
            returns.append(episode_return)
            self._log_test_episode(episode, episode_return)
        self._writer.add_summary('test/returns', np.mean(returns), np.std(returns))

    @property
    def frame(self):
        return self._frame

    @property
    def episode(self):
        return self._episode

    def _done(self, frames, episodes):
        return self._frame > frames or self._episode > episodes

    def _run_training_episode(self):
        # initialize timer
        start_time = timer()
        start_frame = self._frame

        # run the episode
        action, returns = self._reset()
        while not self._env.done:
            self._frame += 1
            action, returns = self._step(action, returns)

        # stop the timer
        end_time = timer()
        fps = (self._frame - start_frame) / (end_time - start_time)

        # log the results
        self._log_training_episode(returns, fps)

        # update experiment state
        self._episode += 1

    def _run_test_episode(self):
        action, returns = self._reset()
        while not self._env.done:
            action, returns = self._step(action, returns)
        return returns

    def _reset(self):
        self._env.reset()
        return self._agent.act(self._env.state, self._env.reward), 0

    def _step(self, action, returns):
        if self._render:
            self._env.render()
        self._env.step(action)
        action = self._agent.act(self._env.state, self._env.reward)
        return action, returns + self._env.reward

    def _make_writer(self, agent_name, env_name, write_loss):
        return ExperimentWriter(self, agent_name, env_name, loss=write_loss)

class ParallelEnvExperiment(Experiment):
    def __init__(
            self,
            agent,
            env,
            render=False,
            quiet=False,
            write_loss=True
    ):
        super().__init__(ExperimentWriter(self, agent.__name__, env.name, loss=write_loss), quiet)
        self._agent = agent(env, self._writer)
        self._env = env
        self._render = render
        self._frame = 1
        self._episode = 1

    def train(self, frames=np.inf, episodes=np.inf):
        pass

    def test(self, episodes=100):
        pass

    @property
    def frame(self):
        return self._frame

    @property
    def episode(self):
        return self._episode

def run_experiment(
        agents,
        envs,
        frames,
        test_episodes=100,
        render=False,
        quiet=False,
        write_loss=True,
):
    if not isinstance(agents, Iterable):
        agents = [agents]

    if not isinstance(envs, Iterable):
        envs = [envs]

    for env in envs:
        for agent in agents:
            make_experiment = get_experiment_type(agent)
            experiment = make_experiment(
                agent,
                env,
                render=render,
                quiet=quiet,
                write_loss=write_loss
            )
            experiment.train(frames=frames)
            experiment.test(episodes=test_episodes)


def get_experiment_type(agent):
    if is_parallel_env_agent(agent):
        return ParallelEnvExperiment
    return SingleEnvExperiment


def is_parallel_env_agent(agent):
    return isinstance(agent, tuple)
