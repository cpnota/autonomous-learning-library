import unittest
from all.presets.validate_agent import validate_agent
from all.presets.tabular import sarsa


class TestSarsa(unittest.TestCase):
    def test_runs(self):
        validate_agent(sarsa, 'FrozenLake-v0')

if __name__ == '__main__':
    unittest.main()
