import unittest
import torch
from all.environments import AtariEnvironment
from all.presets.validate_agent import validate_agent
from all.presets.atari import a2c, dqn, rainbow, reinforce


CPU = torch.device('cpu')
CUDA = torch.device('cuda')


class TestAtariPresets(unittest.TestCase):
    def test_a2c(self):
        validate_agent(a2c(device=CPU),
                       AtariEnvironment('Breakout', device=CPU))

    def test_a2c_cuda(self):
        validate_agent(a2c(device=CUDA),
                       AtariEnvironment('Breakout', device=CUDA))

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
