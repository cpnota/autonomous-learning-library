import unittest

import numpy as np
import torch

from all.environments import GymEnvironment
from all.experiments import SingleEnvExperiment
from all.logging import Logger
from all.presets.classic_control import dqn


class MockLogger(Logger):
    def __init__(self, experiment, label, verbose):
        self.data = {}
        self.hparams = []
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

    def add_summary(self, name, values, step="frame"):
        self._add_scalar("summary/" + name + "/mean", np.mean(values), step)
        self._add_scalar("summary/" + name + "/std", np.std(values), step)

    def add_hparams(self, hparam_dict, metric_dict, step="frame"):
        self.hparams.append((hparam_dict, metric_dict, step))

    def _get_step(self, _type):
        if _type == "frame":
            return self.experiment.frame
        if _type == "episode":
            return self.experiment.episode
        return _type

    def close(self):
        pass


class MockExperiment(SingleEnvExperiment):
    def _make_logger(self, logdir, agent_name, env_name, verbose):
        self._logger = MockLogger(self, agent_name + "_" + env_name, verbose)
        return self._logger


class TestSingleEnvExperiment(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.env = GymEnvironment("CartPole-v0")
        self.env.reset(seed=0)
        self.experiment = None

    def test_adds_default_name(self):
        experiment = MockExperiment(self.make_preset(), self.env, quiet=True)
        self.assertEqual(experiment._logger.label, "dqn_CartPole-v0")

    def test_adds_custom_name(self):
        experiment = MockExperiment(
            self.make_preset(), self.env, name="dqn", quiet=True
        )
        self.assertEqual(experiment._logger.label, "dqn_CartPole-v0")

    def test_writes_training_returns_frame(self):
        experiment = MockExperiment(self.make_preset(), self.env, quiet=True)
        experiment.train(episodes=3)
        np.testing.assert_equal(
            experiment._logger.data["eval/returns"]["values"],
            np.array([22.0, 17.0, 28.0]),
        )
        np.testing.assert_equal(
            experiment._logger.data["eval/returns"]["steps"],
            np.array([23, 40, 68]),
        )

    def test_writes_training_episode_length(self):
        experiment = MockExperiment(self.make_preset(), self.env, quiet=True)
        experiment.train(episodes=3)
        np.testing.assert_equal(
            experiment._logger.data["eval/episode_length"]["values"],
            np.array([22, 17, 28]),
        )
        np.testing.assert_equal(
            experiment._logger.data["eval/episode_length"]["steps"],
            np.array([23, 40, 68]),
        )

    def test_writes_hparams(self):
        experiment = MockExperiment(self.make_preset(), self.env, quiet=True)
        experiment.train(episodes=5)
        returns = experiment.test(episodes=4)
        hparam_dict, metric_dict, step = experiment._logger.hparams[0]
        self.assertDictEqual(hparam_dict, experiment._preset.hyperparameters)
        self.assertEqual(step, "frame")

    def test_writes_test_returns(self):
        experiment = MockExperiment(self.make_preset(), self.env, quiet=True)
        experiment.train(episodes=5)
        returns = experiment.test(episodes=4)
        expected_mean = 8.5
        np.testing.assert_equal(np.mean(returns), expected_mean)
        hparam_dict, metric_dict, step = experiment._logger.hparams[0]
        np.testing.assert_equal(
            metric_dict["test/returns/mean"],
            np.array([expected_mean]),
        )
        np.testing.assert_equal(
            metric_dict["test/returns/std"],
            np.array([0.5]),
        )
        np.testing.assert_equal(
            metric_dict["test/returns/max"],
            np.array([9.0]),
        )
        np.testing.assert_equal(
            metric_dict["test/returns/min"],
            np.array([8.0]),
        )

    def test_writes_test_episode_length(self):
        experiment = MockExperiment(self.make_preset(), self.env, quiet=True)
        experiment.train(episodes=5)
        returns = experiment.test(episodes=4)
        expected_mean = 8.5
        np.testing.assert_equal(np.mean(returns), expected_mean)
        hparam_dict, metric_dict, step = experiment._logger.hparams[0]
        np.testing.assert_equal(
            metric_dict["test/episode_length/mean"],
            np.array([expected_mean]),
        )
        np.testing.assert_equal(
            metric_dict["test/episode_length/std"],
            np.array([0.5]),
        )
        np.testing.assert_equal(
            metric_dict["test/episode_length/max"],
            np.array([9.0]),
        )
        np.testing.assert_equal(
            metric_dict["test/episode_length/min"],
            np.array([8.0]),
        )

    def test_writes_loss(self):
        experiment = MockExperiment(
            self.make_preset(), self.env, quiet=True, verbose=True
        )
        self.assertTrue(experiment._logger.verbose)
        experiment = MockExperiment(
            self.make_preset(), self.env, quiet=True, verbose=False
        )
        self.assertFalse(experiment._logger.verbose)

    def make_preset(self):
        return dqn.device("cpu").env(self.env).build()


if __name__ == "__main__":
    unittest.main()
