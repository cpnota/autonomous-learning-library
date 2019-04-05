import unittest
import torch
from all.environments import AtariEnvironment
from all.presets.validate_agent import validate_agent
from all.presets.atari import dqn, rainbow, reinforce


cpu = torch.device('cpu')
cuda = torch.device('cuda')

class TestAtariPresets(unittest.TestCase):
    def test_dqn(self):
        validate_agent(dqn(replay_start_size=64, device=cpu), AtariEnvironment('Breakout', device=cpu))

    def test_dqn_cuda(self):
        validate_agent(dqn(replay_start_size=64, device=cuda), AtariEnvironment('Breakout', device=cuda))

    def test_rainbow(self):
        validate_agent(rainbow(replay_start_size=64, device=cpu), AtariEnvironment('Breakout', device=cpu))

    def test_rainbow_cuda(self):
        validate_agent(rainbow(replay_start_size=64, device=cuda), AtariEnvironment('Breakout', device=cuda))

    def test_reinforce(self):
        validate_agent(reinforce(device=cpu), AtariEnvironment('Breakout', device=cpu))

    def test_reinforce_cuda(self):
        validate_agent(reinforce(device=cuda), AtariEnvironment('Breakout', device=cuda))

if __name__ == '__main__':
    unittest.main()
