import unittest
from all.environments import AtariEnvironment
from all.presets.validate_agent import validate_agent
from all.presets.reinforce import reinforce_atari


class TestReinforceAtari(unittest.TestCase):
    def test_pong(self):
        validate_agent(reinforce_atari(), AtariEnvironment('Pong'))


if __name__ == '__main__':
    unittest.main()
