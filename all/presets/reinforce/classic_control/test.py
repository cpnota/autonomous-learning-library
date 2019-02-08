import unittest
from all.presets.validate_agent import validate_agent
from all.presets.actor_critic import ac_cc


class TestActorCriticClassicControl(unittest.TestCase):
    def test_cartpole(self):
        validate_agent(ac_cc(), 'CartPole-v0')


if __name__ == '__main__':
    unittest.main()
