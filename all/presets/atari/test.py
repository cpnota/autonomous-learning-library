import unittest
from all.environments import AtariEnvironment
from all.presets.validate_agent import validate_agent
from all.presets.atari import dqn, rainbow, reinforce


class TestAtariPresets(unittest.TestCase):
    def test_dqn(self):
        validate_agent(dqn(replay_start_size=64), AtariEnvironment('Breakout'))

    def test_rainbow(self):
        validate_agent(rainbow(replay_start_size=64), AtariEnvironment('Breakout'))

    def test_reinforce(self):
        validate_agent(reinforce(), AtariEnvironment('Breakout'))

if __name__ == '__main__':
    unittest.main()
