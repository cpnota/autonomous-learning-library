import unittest
import numpy as np
import torch
from all.presets.classic_control import a2c
from all.environments import GymEnvironment
from all.experiments import ParallelEnvExperiment
from single_env_experiment_test import MockWriter

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
        self.env.seed(0)
        self.experiment = None

    def test_adds_label(self):
        experiment = MockExperiment(a2c(), self.env, quiet=True)
        self.assertEqual(experiment._writer.label, "_a2c_CartPole-v0")

    def test_writes_training_returns_eps(self):
        experiment = MockExperiment(a2c(), self.env, quiet=True)
        experiment.train(episodes=3)
        np.testing.assert_equal(
            experiment._writer.data["evaluation/returns/episode"]["steps"],
            np.array([1, 2, 3]),
        )
        np.testing.assert_equal(
            experiment._writer.data["evaluation/returns/episode"]["values"],
            np.array([11., 12., 14]),
        )

    def test_writes_test_returns(self):
        experiment = MockExperiment(a2c(), self.env, quiet=True)
        experiment.train(episodes=5)
        experiment.test(episodes=3)
        np.testing.assert_equal(
            experiment._writer.data["evaluation/test/returns/mean"]["values"],
            np.array([23.]),
        )
        np.testing.assert_equal(
            experiment._writer.data["evaluation/test/returns/std"]["steps"],
            np.array([97]),
        )

    def test_writes_loss(self):
        experiment = MockExperiment(a2c(), self.env, quiet=True, write_loss=True)
        self.assertTrue(experiment._writer.write_loss)
        experiment = MockExperiment(a2c(), self.env, quiet=True, write_loss=False)
        self.assertFalse(experiment._writer.write_loss)

if __name__ == "__main__":
    unittest.main()
