import unittest
import numpy as np
import torch
from all.presets.classic_control import dqn
from all.environments import GymEnvironment
from all.experiments import SingleEnvExperiment
from all.logging import Logger


class MockLogger(Logger):
    def __init__(self, experiment, label, verbose):
        self.data = {}
        self.label = label
        self.verbose = verbose
        self.experiment = experiment

    def _add_scalar(self, key, value, step="frame"):
        if key not in self.data:
            self.data[key] = {"values": [], "steps": []}
        self.data[key]["values"].append(value)
        self.data[key]["steps"].append(self._get_step(step))

    def add_loss(self, name, value, step="frame"):
        pass

    def add_eval(self, name, value, step="frame"):
        self._add_scalar("eval/" + name, value, step)

    def add_info(self, name, value, step="frame"):
        pass

    def add_schedule(self, name, value, step="frame"):
        pass

    def add_summary(self, name, mean, std, step="frame"):
        self._add_scalar("summary/" + name + "/mean", mean, step)
        self._add_scalar("summary/" + name + "/std", std, step)

    def _get_step(self, _type):
        if _type == "frame":
            return self.experiment.frame
        if _type == "episode":
            return self.experiment.episode
        return _type

    def close(self):
        pass


class MockExperiment(SingleEnvExperiment):
    def _make_logger(self, logdir, agent_name, env_name, verbose, logger):
        self._logger = MockLogger(self, agent_name + '_' + env_name, verbose)
        return self._logger


class TestSingleEnvExperiment(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.env = GymEnvironment('CartPole-v0')
        self.env.seed(0)
        self.experiment = None

    def test_adds_default_name(self):
        experiment = MockExperiment(self.make_preset(), self.env, quiet=True)
        self.assertEqual(experiment._logger.label, "dqn_CartPole-v0")

    def test_adds_custom_name(self):
        experiment = MockExperiment(self.make_preset(), self.env, name='dqn', quiet=True)
        self.assertEqual(experiment._logger.label, "dqn_CartPole-v0")

    def test_writes_training_returns_eps(self):
        experiment = MockExperiment(self.make_preset(), self.env, quiet=True)
        experiment.train(episodes=3)
        np.testing.assert_equal(
            experiment._logger.data["eval/returns/episode"]["values"],
            np.array([18., 23., 27.]),
        )
        np.testing.assert_equal(
            experiment._logger.data["eval/returns/episode"]["steps"],
            np.array([1, 2, 3]),
        )

    def test_writes_test_returns(self):
        experiment = MockExperiment(self.make_preset(), self.env, quiet=True)
        experiment.train(episodes=5)
        returns = experiment.test(episodes=4)
        expected_mean = 8.75
        expected_std = 0.433013
        np.testing.assert_equal(np.mean(returns), expected_mean)
        np.testing.assert_equal(
            experiment._logger.data["summary/returns-test/mean"]["values"],
            np.array([expected_mean]),
        )
        np.testing.assert_approx_equal(
            np.array(experiment._logger.data["summary/returns-test/std"]["values"]),
            np.array([expected_std]),
            significant=4
        )
        np.testing.assert_equal(
            experiment._logger.data["summary/returns-test/mean"]["steps"],
            np.array([94]),
        )

    def test_writes_loss(self):
        experiment = MockExperiment(self.make_preset(), self.env, quiet=True, verbose=True)
        self.assertTrue(experiment._logger.verbose)
        experiment = MockExperiment(self.make_preset(), self.env, quiet=True, verbose=False)
        self.assertFalse(experiment._logger.verbose)

    def make_preset(self):
        return dqn.device('cpu').env(self.env).build()


if __name__ == "__main__":
    unittest.main()
