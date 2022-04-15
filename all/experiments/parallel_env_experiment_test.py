import unittest
import numpy as np
import torch
from all.presets.classic_control import a2c
from all.environments import GymEnvironment
from all.experiments import ParallelEnvExperiment
from all.experiments.single_env_experiment_test import MockLogger


class MockExperiment(ParallelEnvExperiment):
    def _make_logger(self, logdir, agent_name, env_name, verbose, logger):
        self._logger = MockLogger(self, agent_name + '_' + env_name, verbose)
        return self._logger


class TestParallelEnvExperiment(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.env = GymEnvironment('CartPole-v0')
        self.env.seed(0)
        self.experiment = MockExperiment(self.make_agent(), self.env, quiet=True)
        self.experiment._env.seed(0)

    def test_adds_default_label(self):
        self.assertEqual(self.experiment._logger.label, "a2c_CartPole-v0")

    def test_adds_custom_label(self):
        env = GymEnvironment('CartPole-v0')
        experiment = MockExperiment(self.make_agent(), env, name='a2c', quiet=True)
        self.assertEqual(experiment._logger.label, "a2c_CartPole-v0")

    def test_writes_training_returns_eps(self):
        self.experiment.train(episodes=3)
        np.testing.assert_equal(
            self.experiment._logger.data["eval/returns/episode"]["steps"],
            np.array([1, 2, 3]),
        )
        np.testing.assert_equal(
            self.experiment._logger.data["eval/returns/episode"]["values"],
            np.array([10., 12., 19.]),
        )

    def test_writes_test_returns(self):
        self.experiment.train(episodes=5)
        returns = self.experiment.test(episodes=4)
        self.assertEqual(len(returns), 4)
        np.testing.assert_equal(
            self.experiment._logger.data["summary/returns-test/mean"]["values"],
            np.array([np.mean(returns)]),
        )
        np.testing.assert_equal(
            self.experiment._logger.data["summary/returns-test/std"]["values"],
            np.array([np.std(returns)]),
        )

    def test_writes_loss(self):
        experiment = MockExperiment(self.make_agent(), self.env, quiet=True, verbose=True)
        self.assertTrue(experiment._logger.verbose)
        experiment = MockExperiment(self.make_agent(), self.env, quiet=True, verbose=False)
        self.assertFalse(experiment._logger.verbose)

    def make_agent(self):
        return a2c.device('cpu').env(self.env).build()


if __name__ == "__main__":
    unittest.main()
