import unittest
from all.environments import GymEnvironment
from all.presets.validate_agent import validate_agent
from all.presets.continuous import ddpg, sac

class TestClassicControlPresets(unittest.TestCase):
    def test_ddpg(self):
        self.validate(ddpg(replay_start_size=50, device='cpu'))

    def test_sac(self):
        self.validate(sac(replay_start_size=50, device='cpu'))

    def validate(self, make_agent):
        validate_agent(make_agent, GymEnvironment('Pendulum-v0'))

if __name__ == '__main__':
    unittest.main()
