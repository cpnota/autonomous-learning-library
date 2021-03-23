import unittest
import numpy as np
import torch
from all.presets.atari import dqn
from all.presets import IndependentMultiagentPreset
from all.environments import MultiagentAtariEnv
from all.experiments import MultiagentEnvExperiment
from all.logging import Writer


class MockWriter(Writer):
    def __init__(self, experiment, label, write_loss):
        self.data = {}
        self.label = label
        self.write_loss = write_loss
        self.experiment = experiment

    def add_scalar(self, key, value, step="frame"):
        if key not in self.data:
            self.data[key] = {"values": [], "steps": []}
        self.data[key]["values"].append(value)
        self.data[key]["steps"].append(self._get_step(step))

    def add_loss(self, name, value, step="frame"):
        pass

    def add_schedule(self, name, value, step="frame"):
        pass

    def add_evaluation(self, name, value, step="frame"):
        self.add_scalar("evaluation/" + name, value, self._get_step(step))

    def add_summary(self, name, mean, std, step="frame"):
        self.add_evaluation(name + "/mean", mean, step)
        self.add_evaluation(name + "/std", std, step)

    def _get_step(self, _type):
        if _type == "frame":
            return self.experiment.frame
        if _type == "episode":
            return self.experiment.episode
        return _type

    def close(self):
        pass


class MockExperiment(MultiagentEnvExperiment):
    def _make_writer(self, logdir, agent_name, env_name, write_loss, writer):
        self._writer = MockWriter(self, agent_name + '_' + env_name, write_loss)
        return self._writer


class TestMultiagentEnvExperiment(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.env = MultiagentAtariEnv('space_invaders_v1', device='cpu')
        self.env.seed(0)
        self.experiment = None

    def test_adds_default_name(self):
        experiment = MockExperiment(self.make_preset(), self.env, quiet=True, save_freq=float('inf'))
        self.assertEqual(experiment._writer.label, "independent_space_invaders_v1")

    def test_adds_custom_name(self):
        experiment = MockExperiment(self.make_preset(), self.env, name='custom', quiet=True, save_freq=float('inf'))
        self.assertEqual(experiment._writer.label, "custom_space_invaders_v1")

    def test_writes_training_returns(self):
        experiment = MockExperiment(self.make_preset(), self.env, quiet=True, save_freq=float('inf'))
        experiment.train(episodes=3)
        self.assertEqual(experiment._writer.data, {
            'evaluation/first_0/returns/frame': {'values': [465.0, 235.0, 735.0, 415.0], 'steps': [766, 1524, 2440, 3038]},
            'evaluation/second_0/returns/frame': {'values': [235.0, 465.0, 170.0, 295.0], 'steps': [766, 1524, 2440, 3038]}
        })

    def test_writes_test_returns(self):
        experiment = MockExperiment(self.make_preset(), self.env, quiet=True, save_freq=float('inf'))
        experiment.train(episodes=3)
        experiment._writer.data = {}
        experiment.test(episodes=3)
        self.assertEqual(list(experiment._writer.data.keys()), [
            'evaluation/first_0/returns-test/mean',
            'evaluation/first_0/returns-test/std',
            'evaluation/second_0/returns-test/mean',
            'evaluation/second_0/returns-test/std'
        ])
        steps = experiment._writer.data['evaluation/first_0/returns-test/mean']['steps'][0]
        for datum in experiment._writer.data.values():
            self.assertEqual(len(datum['values']), 1)
            self.assertGreaterEqual(datum['values'][0], 0.0)
            self.assertEqual(len(datum['steps']), 1)
            self.assertEqual(datum['steps'][0], steps)

    def test_writes_loss(self):
        experiment = MockExperiment(self.make_preset(), self.env, quiet=True, write_loss=True, save_freq=float('inf'))
        self.assertTrue(experiment._writer.write_loss)
        experiment = MockExperiment(self.make_preset(), self.env, quiet=True, write_loss=False, save_freq=float('inf'))
        self.assertFalse(experiment._writer.write_loss)

    def make_preset(self):
        return IndependentMultiagentPreset('independent', 'cpu', {
            agent: dqn.device('cpu').env(env).build()
            for agent, env in self.env.subenvs.items()
        })


if __name__ == "__main__":
    unittest.main()
