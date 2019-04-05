import unittest
import torch
from all.environments import AtariEnvironment
from all.presets.validate_agent import validate_agent
from all.presets.atari import dqn, rainbow, reinforce


CPU = torch.device('CPU')
CUDA = torch.device('CUDA')


class TestAtariPresets(unittest.TestCase):
    def test_dqn(self):
        validate_agent(dqn(replay_start_size=64, device=CPU),
                       AtariEnvironment('Breakout', device=CPU))

    def test_dqn_cuda(self):
        validate_agent(dqn(replay_start_size=64, device=CUDA),
                       AtariEnvironment('Breakout', device=CUDA))

    def test_rainbow(self):
        validate_agent(rainbow(replay_start_size=64, device=CPU),
                       AtariEnvironment('Breakout', device=CPU))

    def test_rainbow_cuda(self):
        validate_agent(rainbow(replay_start_size=64, device=CUDA),
                       AtariEnvironment('Breakout', device=CUDA))

    def test_reinforce(self):
        validate_agent(reinforce(device=CPU),
                       AtariEnvironment('Breakout', device=CPU))

    def test_reinforce_cuda(self):
        validate_agent(reinforce(device=CUDA),
                       AtariEnvironment('Breakout', device=CUDA))


if __name__ == '__main__':
    unittest.main()
