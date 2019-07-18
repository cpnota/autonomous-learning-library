import unittest
from all.environments import GymEnvironment
from all.presets.validate_agent import validate_agent
from all.presets.classic_control import a2c, actor_critic, dqn, ppo, rainbow, vpg, sarsa

class TestClassicControlPresets(unittest.TestCase):
    def test_a2c_(self):
        self.validate(a2c())

    def test_actor_critic(self):
        self.validate(actor_critic())

    def test_dqn(self):
        self.validate(dqn())

    def test_ppo(self):
        self.validate(ppo())

    def test_rainbow(self):
        self.validate(rainbow())

    def test_vpg(self):
        self.validate(vpg())

    def test_sarsa(self):
        self.validate(sarsa())

    def validate(self, make_agent):
        validate_agent(make_agent, GymEnvironment('CartPole-v0'))


if __name__ == '__main__':
    unittest.main()
