import unittest
from all.environments import GymEnvironment
from all.presets.validate_agent import validate_agent
from all.presets.classic_control import actor_critic, dqn, rainbow, reinforce, sarsa

class TestClassicControlPresets(unittest.TestCase):
    def test_actor_critic(self):
        validate_agent(actor_critic(), GymEnvironment('CartPole-v0'))

    def test_dqn(self):
        validate_agent(dqn(), GymEnvironment('CartPole-v0'))

    def test_rainbow(self):
        validate_agent(rainbow(), GymEnvironment('CartPole-v0'))

    def test_reinforce(self):
        validate_agent(reinforce(), GymEnvironment('CartPole-v0'))

    def test_sarsa(self):
        validate_agent(sarsa(), GymEnvironment('CartPole-v0'))


if __name__ == '__main__':
    unittest.main()
