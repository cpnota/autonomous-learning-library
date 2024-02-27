import unittest

import numpy as np
import torch

from all.environments import GymEnvironment
from all.experiments import ParallelEnvExperiment
from all.experiments.single_env_experiment_test import MockLogger
from all.presets.classic_control import a2c


class MockExperiment(ParallelEnvExperiment):
    def _make_logger(self, logdir, agent_name, env_name, verbose):
        self._logger = MockLogger(self, agent_name + "_" + env_name, verbose)
        return self._logger


class TestParallelEnvExperiment(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.env = GymEnvironment("CartPole-v0")
        self.env.reset(seed=0)
        self.experiment = MockExperiment(self.make_agent(), self.env, quiet=True)
        self.experiment._env.reset(seed=0)

    def test_adds_default_label(self):
        self.assertEqual(self.experiment._logger.label, "a2c_CartPole-v0")

    def test_adds_custom_label(self):
        env = GymEnvironment("CartPole-v0")
        experiment = MockExperiment(self.make_agent(), env, name="a2c", quiet=True)
        self.assertEqual(experiment._logger.label, "a2c_CartPole-v0")

    def test_writes_training_returns_frame(self):
        self.experiment.train(episodes=4)
        np.testing.assert_equal(
            self.experiment._logger.data["eval/returns"]["steps"],
            np.array([65, 65, 101, 125]),
        )
        np.testing.assert_equal(
            self.experiment._logger.data["eval/returns"]["values"],
            np.array([16.0, 16.0, 25.0, 14.0]),
        )

    def test_writes_training_episode_length(self):
        self.experiment.train(episodes=4)
        np.testing.assert_equal(
            self.experiment._logger.data["eval/episode_length"]["steps"],
            np.array([65, 65, 101, 125]),
        )
        np.testing.assert_equal(
            self.experiment._logger.data["eval/episode_length"]["values"],
            np.array([16.0, 16.0, 25.0, 14.0]),
        )

    def test_writes_hparams(self):
        experiment = self.experiment
        experiment.train(episodes=5)
        returns = experiment.test(episodes=4)
        hparam_dict, metric_dict, step = experiment._logger.hparams[0]
        self.assertDictEqual(hparam_dict, experiment._preset.hyperparameters)
        self.assertEqual(step, "frame")

    def test_writes_test_returns(self):
        experiment = self.experiment
        experiment.train(episodes=5)
        returns = experiment.test(episodes=4)
        expected_mean = 26.25
        np.testing.assert_equal(np.mean(returns), expected_mean)
        hparam_dict, metric_dict, step = experiment._logger.hparams[0]
        np.testing.assert_equal(
            metric_dict["test/returns/mean"],
            np.array([expected_mean]),
        )
        np.testing.assert_almost_equal(
            metric_dict["test/returns/std"], np.array([6.869]), decimal=3
        )
        np.testing.assert_equal(
            metric_dict["test/returns/max"],
            np.array([34.0]),
        )
        np.testing.assert_equal(
            metric_dict["test/returns/min"],
            np.array([18.0]),
        )

    def test_writes_test_episode_length(self):
        experiment = self.experiment
        experiment.train(episodes=5)
        returns = experiment.test(episodes=4)
        expected_mean = 26.25
        np.testing.assert_equal(np.mean(returns), expected_mean)
        hparam_dict, metric_dict, step = experiment._logger.hparams[0]
        np.testing.assert_equal(
            metric_dict["test/episode_length/mean"],
            np.array([expected_mean]),
        )
        np.testing.assert_almost_equal(
            metric_dict["test/episode_length/std"], np.array([6.869]), decimal=3
        )
        np.testing.assert_equal(
            metric_dict["test/episode_length/max"],
            np.array([34.0]),
        )
        np.testing.assert_equal(
            metric_dict["test/episode_length/min"],
            np.array([18.0]),
        )

    def test_writes_loss(self):
        experiment = MockExperiment(
            self.make_agent(), self.env, quiet=True, verbose=True
        )
        self.assertTrue(experiment._logger.verbose)
        experiment = MockExperiment(
            self.make_agent(), self.env, quiet=True, verbose=False
        )
        self.assertFalse(experiment._logger.verbose)

    def make_agent(self):
        return a2c.device("cpu").env(self.env).build()


if __name__ == "__main__":
    unittest.main()
