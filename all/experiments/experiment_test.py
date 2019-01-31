import unittest
import os
import numpy as np
import torch
from all.presets.tabular import sarsa
from all.presets.tabular import actor_critic
from . import Experiment


class TestExperiment(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.experiment = Experiment('NChain-v0', episodes=3, trials=2)
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
    experiment.run(sarsa)
    experiment.run(actor_critic)
    return experiment


def check_results(results):
    np.testing.assert_equal(
        results["data"]["sarsa"],
        np.array([[1518., 1546., 1530.],
                  [1860., 1262., 1552.]])
    )
    np.testing.assert_equal(
        results["data"]["actor_critic"],
        np.array([[1280., 1370., 1492.],
                  [1384., 1414., 1434.]])
    )


if __name__ == '__main__':
    unittest.main()
