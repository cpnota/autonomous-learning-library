import unittest
from all.environments import GymEnvironment
from all.presets.validate_agent import validate_agent
from all.presets.continuous import actor_critic

class TestClassicControlPresets(unittest.TestCase):
    def test_actor_critic(self):
        self.validate(actor_critic(device='cpu'))

    def validate(self, make_agent):
        validate_agent(make_agent, GymEnvironment('Pendulum-v0'))

if __name__ == '__main__':
    unittest.main()
