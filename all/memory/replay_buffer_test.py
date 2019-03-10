import unittest
import random
import torch
import numpy as np
from all.memory import ReplayBuffer

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        random.seed(1)
        np.random.seed(1)
        self.replay_buffer = ReplayBuffer(5)

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


if __name__ == '__main__':
    unittest.main()
