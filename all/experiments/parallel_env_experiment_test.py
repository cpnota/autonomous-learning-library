import unittest
import numpy as np
import torch
from all.presets.classic_control import a2c
from all.environments import GymEnvironment
from all.experiments import ParallelEnvExperiment
from all.experiments.single_env_experiment_test import MockWriter

# pylint: disable=protected-access
class MockExperiment(ParallelEnvExperiment):
    def _make_writer(self, agent_name, env_name, write_loss):
        self._writer = MockWriter(self, agent_name + '_' +  env_name, write_loss)
        return self._writer


class TestParalleleEnvExperiment(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.env = GymEnvironment('CartPole-v0')
        self.experiment = MockExperiment(a2c(), self.env, quiet=True)
        for i, env in enumerate(self.experiment._env):
            env.seed(i)

    def test_adds_label(self):
        self.assertEqual(self.experiment._writer.label, "_a2c_CartPole-v0")

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
        self.experiment.test(episodes=3)
        np.testing.assert_equal(
            self.experiment._writer.data["evaluation/test/returns/mean"]["values"],
            np.array([16.]),
        )
        np.testing.assert_equal(
            self.experiment._writer.data["evaluation/test/returns/std"]["steps"],
            np.array([196]),
        )

    def test_writes_loss(self):
        experiment = MockExperiment(a2c(), self.env, quiet=True, write_loss=True)
        self.assertTrue(experiment._writer.write_loss)
        experiment = MockExperiment(a2c(), self.env, quiet=True, write_loss=False)
        self.assertFalse(experiment._writer.write_loss)

if __name__ == "__main__":
    unittest.main()
