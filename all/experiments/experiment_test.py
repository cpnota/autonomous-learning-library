import unittest
import numpy as np
import torch
from all.presets.classic_control import sarsa
from all.experiments import Experiment

class MockWriter():
    def __init__(self, label):
        self.data = {}
        self.label = label

    def add_scalar(self, key, value, step):
        if not key in self.data:
            self.data[key] = {
                "values": [],
                "steps": []
            }
        self.data[key]["values"].append(value)
        self.data[key]["steps"].append(step)

class MockExperiment(Experiment):
    def _make_writer(self, label):
        return MockWriter(label)

# pylint: disable=protected-access
class TestExperiment(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.experiment = MockExperiment('CartPole-v0', episodes=3)
        self.experiment.env.seed(0)

    def test_adds_label(self):
        self.experiment.run(sarsa(), console=False)
        self.assertEqual(self.experiment._writer.label, "_sarsa")

    def test_writes_returns_eps(self):
        self.experiment.run(sarsa(), console=False)
        np.testing.assert_equal(
            self.experiment._writer.data["CartPole-v0/returns/eps"]["values"],
            np.array([9., 11., 10.])
        )
        np.testing.assert_equal(
            self.experiment._writer.data["CartPole-v0/returns/eps"]["steps"],
            np.array([1, 2, 3])
        )

if __name__ == '__main__':
    unittest.main()
