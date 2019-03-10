import unittest
from all.environments import GymEnvironment
from all.presets.validate_agent import validate_agent
from all.presets.dqn import dqn


class TestDqnAtari(unittest.TestCase):
    def test_pong(self):
        validate_agent(dqn(), GymEnvironment('PongNoFrameskip-v4'))


if __name__ == '__main__':
    unittest.main()
