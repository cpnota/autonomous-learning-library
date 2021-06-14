import unittest
import numpy as np
import torch
from all.presets.classic_control import a2c
from all.environments import GymEnvironment
from all.experiments import ParallelEnvExperiment
from all.experiments.single_env_experiment_test import MockWriter


class MockExperiment(ParallelEnvExperiment):
    def _make_writer(self, logdir, agent_name, env_name, write_loss, writer):
        self._writer = MockWriter(self, agent_name + '_' + env_name, write_loss)
        return self._writer


class TestParallelEnvExperiment(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.env = GymEnvironment('CartPole-v0')
        self.env.seed(0)
        self.experiment = MockExperiment(self.make_agent(), self.env, quiet=True)
        self.experiment._env.seed(0)

    def test_adds_default_label(self):
        self.assertEqual(self.experiment._writer.label, "a2c_CartPole-v0")

    def test_adds_custom_label(self):
        env = GymEnvironment('CartPole-v0')
        experiment = MockExperiment(self.make_agent(), env, name='a2c', quiet=True)
        self.assertEqual(experiment._writer.label, "a2c_CartPole-v0")

    def test_writes_training_returns_eps(self):
        self.experiment.train(episodes=3)
        np.testing.assert_equal(
            self.experiment._writer.data["evaluation/returns/episode"]["steps"],
            np.array([1, 2, 3]),
        )
        np.testing.assert_equal(
            self.experiment._writer.data["evaluation/returns/episode"]["values"],
            np.array([10., 11., 17.]),
        )

    def test_writes_test_returns(self):
        self.experiment.train(episodes=5)
        returns = self.experiment.test(episodes=4)
        self.assertEqual(len(returns), 4)
        np.testing.assert_equal(
            self.experiment._writer.data["evaluation/returns-test/mean"]["values"],
            np.array([np.mean(returns)]),
        )
        np.testing.assert_equal(
            self.experiment._writer.data["evaluation/returns-test/std"]["values"],
            np.array([np.std(returns)]),
        )

    def test_writes_loss(self):
        experiment = MockExperiment(self.make_agent(), self.env, quiet=True, write_loss=True)
        self.assertTrue(experiment._writer.write_loss)
        experiment = MockExperiment(self.make_agent(), self.env, quiet=True, write_loss=False)
        self.assertFalse(experiment._writer.write_loss)

    def make_agent(self):
        return a2c.device('cpu').env(self.env).build()


if __name__ == "__main__":
    unittest.main()
