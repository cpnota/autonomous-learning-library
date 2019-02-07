import unittest
from all.presets.validate_agent import validate_agent
from all.presets.dqn import dqn_cc


class TestDqnClassicControl(unittest.TestCase):
    def test_cartpole(self):
        validate_agent(dqn_cc(), 'CartPole-v0')


if __name__ == '__main__':
    unittest.main()
