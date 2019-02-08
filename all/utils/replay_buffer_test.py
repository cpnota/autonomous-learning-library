import unittest
import random
import torch
import numpy
from all.utils import ReplayBuffer

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        random.seed(1)
        self.replay_buffer = ReplayBuffer(5)

    def test_run(self):
        states = torch.arange(0, 20)
        actions = torch.arange(0, 20)
        rewards = torch.arange(0, 20)
        expected = [
            [0, 0, 0],
            [0, 1, 1],
            [1, 2, 1],
            [1, 0, 3],
            [0, 3, 3],
            [5, 1, 4],
            [4, 3, 6],
            [3, 5, 3],
            [4, 4, 8],
            [5, 8, 6]
        ]
        for i in range(10):
            self.replay_buffer.store(states[i], actions[i], states[i+1], rewards[i])
            sample = self.replay_buffer.sample(3)
            sample_states = torch.tensor(sample[0]).detach().numpy()
            numpy.testing.assert_array_equal(sample_states, expected[i])


if __name__ == '__main__':
    unittest.main()
