import unittest
from all.environments import make_atari
from all.presets.validate_agent import validate_agent
from all.presets.reinforce import reinforce_atari


class TestReinforceAtari(unittest.TestCase):
    def test_pong(self):
        validate_agent(reinforce_atari(), make_atari('PongNoFrameskip-v4'))


if __name__ == '__main__':
    unittest.main()
