from abc import ABC, abstractmethod
import numpy as np


class Experiment(ABC):
    '''
    An Experiment manages the basic train/test loop and logs results.

    Args:
            logger (:torch.logging.logger:): A Logger object used for logging.
            quiet (bool): If False, the Experiment will print information about
                episode returns to standard out.
    '''

    def __init__(self, logger, quiet):
        self._logger = logger
        self._quiet = quiet
        self._best_returns = -np.inf
        self._returns100 = []

    @abstractmethod
    def train(self, frames=np.inf, episodes=np.inf):
        '''
        Train the agent for a certain number of frames or episodes.
        If both frames and episodes are specified, then the training loop will exit
        when either condition is satisfied.

        Args:
                frames (int): The maximum number of training frames.
                episodes (bool): The maximum number of training episodes.
        '''

    @abstractmethod
    def test(self, episodes=100):
        '''
        Test the agent in eval mode for a certain number of episodes.

        Args:
            episodes (int): The number of test episodes.

        Returns:
            list(float): A list of all returns received during testing.
        '''

    @property
    @abstractmethod
    def frame(self):
        '''The index of the current training frame.'''

    @property
    @abstractmethod
    def episode(self):
        '''The index of the current training episode'''

    def _log_training_episode(self, returns, fps):
        if not self._quiet:
            print('episode: {}, frame: {}, fps: {}, returns: {}'.format(self.episode, self.frame, int(fps), returns))
        if returns > self._best_returns:
            self._best_returns = returns
        self._returns100.append(returns)
        if len(self._returns100) == 100:
            mean = np.mean(self._returns100)
            std = np.std(self._returns100)
            self._logger.add_summary('returns100', mean, std, step="frame")
            self._returns100 = []
        self._logger.add_eval('returns/episode', returns, step="episode")
        self._logger.add_eval('returns/frame', returns, step="frame")
        self._logger.add_eval("returns/max", self._best_returns, step="frame")
        self._logger.add_eval('fps', fps, step="frame")

    def _log_test_episode(self, episode, returns):
        if not self._quiet:
            print('test episode: {}, returns: {}'.format(episode, returns))

    def _log_test(self, returns):
        if not self._quiet:
            mean = np.mean(returns)
            sem = np.var(returns) / np.sqrt(len(returns))
            print('test returns (mean ± sem): {} ± {}'.format(mean, sem))
        self._logger.add_summary('returns-test', np.mean(returns), np.std(returns))

    def save(self):
        return self._preset.save('{}/preset.pt'.format(self._logger.log_dir))

    def close(self):
        self._logger.close()
