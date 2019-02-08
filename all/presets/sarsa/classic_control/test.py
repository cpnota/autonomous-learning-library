import unittest
from all.presets.validate_agent import validate_agent
from all.presets.sarsa import sarsa_cc


class TestSarsaClassicControl(unittest.TestCase):
    def test_cartpole(self):
        validate_agent(sarsa_cc(), 'CartPole-v0')


if __name__ == '__main__':
    unittest.main()
