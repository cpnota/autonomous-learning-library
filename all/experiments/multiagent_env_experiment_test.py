import random
import unittest

import numpy as np
import torch

from all.environments import MultiagentAtariEnv
from all.experiments import MultiagentEnvExperiment
from all.experiments.single_env_experiment_test import MockLogger
from all.presets import IndependentMultiagentPreset
from all.presets.atari import dqn


class MockExperiment(MultiagentEnvExperiment):
    def _make_logger(self, logdir, agent_name, env_name, verbose):
        self._logger = MockLogger(self, agent_name + "_" + env_name, verbose)
        return self._logger


class TestMultiagentEnvExperiment(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        self.env = MultiagentAtariEnv("space_invaders_v2", device="cpu", seed=0)
        self.env.reset(seed=0)
        self.experiment = None

    def test_adds_default_name(self):
        experiment = MockExperiment(
            self.make_preset(), self.env, quiet=True, save_freq=float("inf")
        )
        self.assertEqual(experiment._logger.label, "independent_space_invaders_v2")

    def test_adds_custom_name(self):
        experiment = MockExperiment(
            self.make_preset(),
            self.env,
            name="custom",
            quiet=True,
            save_freq=float("inf"),
        )
        self.assertEqual(experiment._logger.label, "custom_space_invaders_v2")

    def test_writes_training_returns(self):
        experiment = MockExperiment(
            self.make_preset(), self.env, quiet=True, save_freq=float("inf")
        )
        experiment.train(episodes=3)
        self.maxDiff = None
        # could not get the exact numbers to be reproducible across enviornments :(
        self.assertEqual(
            len(experiment._logger.data["eval/first_0/returns"]["values"]), 3
        )
        self.assertEqual(
            len(experiment._logger.data["eval/first_0/returns"]["steps"]), 3
        )
        self.assertEqual(
            len(experiment._logger.data["eval/second_0/returns"]["values"]), 3
        )
        self.assertEqual(
            len(experiment._logger.data["eval/second_0/returns"]["steps"]), 3
        )

    def test_writes_test_returns(self):
        experiment = MockExperiment(
            self.make_preset(), self.env, quiet=True, save_freq=float("inf")
        )
        experiment.train(episodes=3)
        experiment._logger.data = {}
        experiment.test(episodes=3)
        self.assertEqual(
            list(experiment._logger.data.keys()),
            [
                "summary/first_0/returns-test/mean",
                "summary/first_0/returns-test/std",
                "summary/second_0/returns-test/mean",
                "summary/second_0/returns-test/std",
            ],
        )
        steps = experiment._logger.data["summary/first_0/returns-test/mean"]["steps"][0]
        for datum in experiment._logger.data.values():
            self.assertEqual(len(datum["values"]), 1)
            self.assertGreaterEqual(datum["values"][0], 0.0)
            self.assertEqual(len(datum["steps"]), 1)
            self.assertEqual(datum["steps"][0], steps)

    def test_writes_loss(self):
        experiment = MockExperiment(
            self.make_preset(),
            self.env,
            quiet=True,
            verbose=True,
            save_freq=float("inf"),
        )
        self.assertTrue(experiment._logger.verbose)
        experiment = MockExperiment(
            self.make_preset(),
            self.env,
            quiet=True,
            verbose=False,
            save_freq=float("inf"),
        )
        self.assertFalse(experiment._logger.verbose)

    def make_preset(self):
        return IndependentMultiagentPreset(
            "independent",
            "cpu",
            {
                agent: dqn.device("cpu").env(env).build()
                for agent, env in self.env.subenvs.items()
            },
        )


if __name__ == "__main__":
    unittest.main()
