import unittest
from all.environments import AtariEnvironment
from all.presets.validate_agent import validate_agent
from all.presets.dqn import dqn


class TestDqnAtari(unittest.TestCase):
    def test_pong(self):
        validate_agent(dqn(), AtariEnvironment('Pong'))


if __name__ == '__main__':
    unittest.main()
