import unittest
import os
import numpy as np
import torch
from all.presets.sarsa import sarsa_cc
from all.presets.actor_critic import ac_cc
from . import Experiment


class TestExperiment(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.experiment = Experiment('CartPole-v0', episodes=3, trials=2)
        self.experiment.env.seed(0)

    def test_run(self):
        run_experiment(self.experiment)
        check_results(self.experiment.results)

    def test_save_and_load(self):
        run_experiment(self.experiment)

        # save and load
        self.experiment.save('test_experiment.json')
        results = Experiment.load('test_experiment.json')
        os.remove('test_experiment.json')

        check_results(results)


def run_experiment(experiment):
    experiment.run(sarsa_cc(), agent_name="sarsa")
    experiment.run(ac_cc(), agent_name="actor_critic")
    return experiment


def check_results(results):
    np.testing.assert_equal(
        results["data"]["sarsa"],
        np.array([[9., 10., 10.], [18., 10., 11.]])
    )
    np.testing.assert_equal(
        results["data"]["actor_critic"],
        np.array([[14., 13., 29.],
                  [14., 32., 10.]])
    )


if __name__ == '__main__':
    unittest.main()
