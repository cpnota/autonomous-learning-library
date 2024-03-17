import unittest
from unittest import mock

import torch

from all.environments import GymEnvironment
from all.experiments.watch import load_and_watch


class MockAgent:
    def act(self):
        # sample from cartpole action space
        return torch.randint(0, 2, [])


class MockPreset:
    def __init__(self, filename):
        self.filename = filename

    def test_agent(self):
        return MockAgent


class WatchTest(unittest.TestCase):
    @mock.patch("torch.load", lambda filename: MockPreset(filename))
    @mock.patch("time.sleep", mock.MagicMock())
    @mock.patch("sys.stdout", mock.MagicMock())
    def test_load_and_watch(self):
        env = mock.MagicMock(GymEnvironment("CartPole-v0", render_mode="rgb_array"))
        load_and_watch("file.name", env, n_episodes=3)
        self.assertEqual(env.reset.call_count, 4)


if __name__ == "__main__":
    unittest.main()
