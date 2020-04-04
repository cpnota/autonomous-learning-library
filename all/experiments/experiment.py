from abc import ABC, abstractmethod
import numpy as np


class Experiment(ABC):
    def __init__(self, writer, quiet):
        self._writer = writer
        self._quiet = quiet
        self._best_returns = -np.inf
        self._returns100 = []

    @property
    @abstractmethod
    def frame(self):
        pass

    @property
    @abstractmethod
    def episode(self):
        pass

    @abstractmethod
    def train(self, frames=np.inf, episodes=np.inf):
        pass

    @abstractmethod
    def test(self, frames=np.inf, episodes=np.inf):
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

    def _log_test(self, returns):
        self._writer.add_summary('returns-test', np.mean(returns), np.std(returns))
