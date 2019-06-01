import unittest
import random
import torch
import numpy as np
import torch_testing as tt
from all.environments import State
from all.memory import ExperienceReplayBuffer, PrioritizedReplayBuffer

class TestExperienceReplayBuffer(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        random.seed(1)
        torch.manual_seed(1)
        self.replay_buffer = ExperienceReplayBuffer(5)

    def test_run(self):
        states = torch.arange(0, 20)
        actions = torch.arange(0, 20)
        rewards = torch.arange(0, 20)
        expected_samples = torch.tensor([
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
            [3, 0, 0],
            [1, 4, 4],
            [1, 2, 4],
            [2, 4, 3],
            [4, 7, 4],
            [7, 4, 6],
            [6, 5, 6]
        ])
        expected_weights = np.ones((10, 3))
        actual_samples = []
        actual_weights = []
        for i in range(10):
            state = State(states[i].unsqueeze(0), torch.tensor([1]))
            next_state = State(states[i + 1].unsqueeze(0), torch.tensor([1]))
            self.replay_buffer.store(
                state, actions[i], next_state, rewards[i])
            sample = self.replay_buffer.sample(3)
            actual_samples.append(sample[0].features)
            actual_weights.append(sample[-1])
        tt.assert_equal(torch.cat(actual_samples).view(expected_samples.shape), expected_samples)
        np.testing.assert_array_equal(expected_weights, np.vstack(actual_weights))

class TestPrioritizedReplayBuffer(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.replay_buffer = PrioritizedReplayBuffer(5, 0.6)

    def test_run(self):
        states = torch.arange(0, 20)
        actions = torch.arange(0, 20)
        rewards = torch.arange(0, 20)
        expected_samples = [
            [0, 2, 2],
            [0, 1, 1],
            [3, 3, 5],
            [5, 3, 6],
            [3, 5, 7],
            [8, 5, 8],
            [8, 5, 5],
        ]
        expected_weights = [[1., 1., 1.],
                            [0.56589746, 0.5124394, 0.5124394],
                            [0.5124343, 0.5124343, 0.5124343],
                            [0.5090894, 0.6456939, 0.46323255],
                            [0.51945686, 0.5801515, 0.45691562],
                            [0.45691025, 0.5096957, 0.45691025],
                            [0.5938914, 0.6220026, 0.6220026]]
        actual_samples = []
        actual_weights = []
        for i in range(10):
            state = State(states[i].unsqueeze(0), torch.tensor([1]))
            next_state = State(states[i + 1].unsqueeze(0), torch.tensor([1]))
            self.replay_buffer.store(
                state, actions[i], next_state, rewards[i])
            if i > 2:
                sample = self.replay_buffer.sample(3)
                sample_states = torch.tensor(sample[0].features).detach().numpy()
                self.replay_buffer.update_priorities(torch.randn(3))
                actual_samples.append(sample_states)
                actual_weights.append(sample[-1])
        np.testing.assert_array_equal(
            expected_samples, np.vstack(actual_samples))
        np.testing.assert_array_almost_equal(
            expected_weights, np.vstack(actual_weights))


if __name__ == '__main__':
    unittest.main()
