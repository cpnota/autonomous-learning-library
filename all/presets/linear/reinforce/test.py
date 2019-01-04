import unittest
from all.presets.validate_agent import validate_agent
from all.presets.linear import reinforce


class TestReinforce(unittest.TestCase):
    def test_runs(self):
        validate_agent(reinforce, 'CartPole-v0')

if __name__ == '__main__':
    unittest.main()
