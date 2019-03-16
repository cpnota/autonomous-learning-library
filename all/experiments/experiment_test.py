import unittest
import os
import numpy as np
import torch
from all.presets.sarsa import sarsa_cc
from all.experiments import Experiment

class MockWriter():
    data = {}

    def add_scalar(self, key, value, step):
        if not key in self.data:
            self.data[key] = {
                "values": [],
                "steps": []
            }
        self.data[key]["values"].append(value)
        self.data[key]["steps"].append(step)

class MockExperiment(Experiment):
    def _make_writer(self):
        return MockWriter()

class TestExperiment(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.experiment = MockExperiment('CartPole-v0', episodes=3, trials=2)
        self.experiment.env.seed(0)

    def test_run(self):
        self.experiment.run(sarsa_cc(), agent_name="sarsa")
        np.testing.assert_equal(
            self.experiment._writer.data["returns"]["values"],
            np.array([9., 10., 10., 18., 10., 11.])
        )
        np.testing.assert_equal(
            self.experiment._writer.data["returns"]["steps"],
            np.array([0, 1, 2, 0, 1, 2])
        )

if __name__ == '__main__':
    unittest.main()
