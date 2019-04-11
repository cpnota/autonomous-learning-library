import unittest
import numpy as np
import torch
from all.presets.classic_control import sarsa
from all.experiments import Experiment, Writer


class MockWriter(Writer):
    def __init__(self, label):
        self.data = {}
        self.label = label
        self.frames = 0
        self.episodes = 1

    def add_scalar(self, key, value, step="frame"):
        if not key in self.data:
            self.data[key] = {
                "values": [],
                "steps": []
            }
        self.data[key]["values"].append(value)
        self.data[key]["steps"].append(self._get_step(step))

    def add_loss(self, name, value, step="frame"):
        pass

    def add_evaluation(self, name, value, step="frame"):
        self.add_scalar('evaluation/' + name, value, self._get_step(step))

    def _get_step(self, _type):
        if _type == "frame":
            return self.frames
        if _type == "episode":
            return self.episodes
        return _type


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
            self.experiment._writer.data["evaluation/returns-by-episode"]["values"],
            np.array([9., 11., 10.])
        )
        np.testing.assert_equal(
            self.experiment._writer.data["evaluation/returns-by-episode"]["steps"],
            np.array([1, 2, 3])
        )


if __name__ == '__main__':
    unittest.main()
