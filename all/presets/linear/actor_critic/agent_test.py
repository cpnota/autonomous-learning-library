import unittest
from all.presets.validate_agent import validate_agent
from all.presets.linear import actor_critic


class TestActorCritic(unittest.TestCase):
    def test_runs(self):
        validate_agent(actor_critic, 'CartPole-v0')

if __name__ == '__main__':
    unittest.main()
