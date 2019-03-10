import unittest
import torch
import random
import numpy as np
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
        expected = [
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
        ]
        actual = []
        for i in range(10):
            self.replay_buffer.store(
                states[i], actions[i], states[i + 1], rewards[i])
            sample = self.replay_buffer.sample(3)
            sample_states = torch.tensor(sample[0]).detach().numpy()
            actual.append(sample_states)
        np.testing.assert_array_equal(expected, np.vstack(actual))

class TestPrioritizedReplayBuffer(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.replay_buffer = PrioritizedReplayBuffer(5, 0.6)

    def test_run(self):
        states = torch.arange(0, 20)
        actions = torch.arange(0, 20)
        rewards = torch.arange(0, 20)
        expected_samples = [
            [0, 0, 0],
            [0, 0, 0],
            [5, 5, 5],
            [5, 6, 5],
            [7, 5, 6],
            [8, 5, 8],
            [8, 5, 5],
        ]
        expected_weights = [[1., 1., 1., ],
                            [1., 1., 1., ],
                            [1., 1., 1., ],
                            [1., 0.9099231, 1., ],
                            [0.45689753, 1., 0.5194369],
                            [0.6109873, 0.6815734, 0.6109873, ],
                            [0.79416645, 0.83175826, 0.83175826]]
        actual_samples = []
        actual_weights = []
        for i in range(10):
            self.replay_buffer.store(
                states[i], actions[i], states[i + 1], rewards[i])
            if i > 2:
                sample, weights = self.replay_buffer.sample(3)
                sample_states = torch.tensor(sample[0]).detach().numpy()
                self.replay_buffer.update_priorities(torch.randn(3))
                actual_samples.append(sample_states)
                actual_weights.append(weights)
        np.testing.assert_array_equal(
            expected_samples, np.vstack(actual_samples))
        np.testing.assert_array_almost_equal(
            expected_weights, np.vstack(actual_weights))


if __name__ == '__main__':
    unittest.main()
