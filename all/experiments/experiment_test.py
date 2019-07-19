import unittest
import numpy as np
import torch
from all.presets.classic_control import dqn
from all.experiments import Experiment, Writer


class MockWriter(Writer):
    def __init__(self, label, write_loss):
        self.data = {}
        self.label = label
        self.frames = 0
        self.episodes = 1
        self.write_loss = write_loss

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
    def _make_writer(self, label, write_loss=True):
        return MockWriter(label, write_loss)

# pylint: disable=protected-access


class TestExperiment(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.experiment = MockExperiment('CartPole-v0', episodes=3)
        self.experiment.env.seed(0)

    def test_adds_label(self):
        self.experiment.run(dqn(), console=False)
        self.assertEqual(self.experiment._writer.label, "_dqn")

    def test_writes_returns_eps(self):
        self.experiment.run(dqn(), console=False)
        np.testing.assert_equal(
            self.experiment._writer.data["evaluation/returns-by-episode"]["values"],
            np.array([14., 19., 26.])
        )
        np.testing.assert_equal(
            self.experiment._writer.data["evaluation/returns-by-episode"]["steps"],
            np.array([1, 2, 3])
        )

    def test_writes_loss(self):
        self.experiment.run(dqn(), console=False)
        self.assertTrue(self.experiment._writer.write_loss)
        self.experiment.run(dqn(), console=False, write_loss=False)
        self.assertFalse(self.experiment._writer.write_loss)

if __name__ == '__main__':
    unittest.main()
