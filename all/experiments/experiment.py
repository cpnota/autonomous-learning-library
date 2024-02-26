from abc import ABC, abstractmethod

import numpy as np


class Experiment(ABC):
    """
    An Experiment manages the basic train/test loop and logs results.

    Args:
            logger (:torch.logging.logger:): A Logger object used for logging.
            quiet (bool): If False, the Experiment will print information about
                episode returns to standard out.
    """

    def __init__(self, logger, quiet):
        self._logger = logger
        self._quiet = quiet
        self._best_returns = -np.inf
        self._returns100 = []

    @abstractmethod
    def train(self, frames=np.inf, episodes=np.inf):
        """
        Train the agent for a certain number of frames or episodes.
        If both frames and episodes are specified, then the training loop will exit
        when either condition is satisfied.

        Args:
                frames (int): The maximum number of training frames.
                episodes (bool): The maximum number of training episodes.
        """

    @abstractmethod
    def test(self, episodes=100):
        """
        Test the agent in eval mode for a certain number of episodes.

        Args:
            episodes (int): The number of test episodes.

        Returns:
            list(float): A list of all returns received during testing.
        """

    @property
    @abstractmethod
    def frame(self):
        """The index of the current training frame."""

    @property
    @abstractmethod
    def episode(self):
        """The index of the current training episode"""

    def _log_training_episode(self, returns, episode_length, fps):
        if not self._quiet:
            print(
                "episode: {}, frame: {}, fps: {}, episode_length: {}, returns: {}".format(
                    self.episode, self.frame, int(fps), episode_length, returns
                )
            )
        if returns > self._best_returns:
            self._best_returns = returns
        self._returns100.append(returns)
        if len(self._returns100) == 100:
            self._logger.add_summary("returns100", self._returns100)
            self._returns100 = []
        self._logger.add_eval("returns", returns)
        self._logger.add_eval("episode_length", episode_length)
        self._logger.add_eval("fps", fps)

    def _log_test_episode(self, episode, returns, episode_length):
        if not self._quiet:
            print(
                "test episode: {}, episode_length: {}, returns: {}".format(
                    episode, episode_length, returns
                )
            )

    def _log_test(self, returns, episode_lengths):
        if not self._quiet:
            returns_mean = np.mean(returns)
            returns_sem = np.std(returns) / np.sqrt(len(returns))
            print(
                "test returns (mean ± sem): {} ± {}".format(returns_mean, returns_sem)
            )
            episode_length_mean = np.mean(episode_lengths)
            episode_length_sem = np.std(episode_lengths) / np.sqrt(len(episode_lengths))
            print(
                "test episode length (mean ± sem): {} ± {}".format(
                    episode_length_mean, episode_length_sem
                )
            )
        metrics = {
            "test/returns": returns,
            "test/episode_length": episode_lengths,
        }
        aggregators = ["mean", "std", "max", "min"]
        metrics_dict = {
            f"{metric}/{aggregator}": getattr(np, aggregator)(values)
            for metric, values in metrics.items()
            for aggregator in aggregators
        }
        self._logger.add_hparams(self._preset.hyperparameters, metrics_dict)

    def save(self):
        return self._preset.save("{}/preset.pt".format(self._logger.log_dir))

    def close(self):
        self._logger.close()
