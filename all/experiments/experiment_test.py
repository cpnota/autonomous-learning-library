import unittest
import os
import numpy as np
from all.presets.tabular import sarsa
from all.presets.tabular import actor_critic
from . import Experiment


class TestExperiment(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
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
        np.array([[1558., 1558., 1526.],
                  [996., 1044., 1042.]])
    )
    np.testing.assert_equal(
        results["data"]["sarsa"],
        np.array([[1558., 1558., 1526.],
                  [996., 1044., 1042.]])
    )


if __name__ == '__main__':
    unittest.main()
