import unittest
from all.environments import GymEnvironment
from all.presets.continuous import ddpg, ppo, sac
from validate_agent import validate_agent


class TestContinuousPresets(unittest.TestCase):
    def test_ddpg(self):
        self.validate(ddpg(replay_start_size=50, device='cpu'))

    def test_ppo(self):
        self.validate(ppo(n_envs=4, n_steps=4, epochs=4, minibatches=4, device='cpu'))

    def test_sac(self):
        validate_agent(
            sac().device('cpu').hyperparameters(replay_start_size=50),
            GymEnvironment('LunarLanderContinuous-v2')
        )

    def validate(self, make_agent):
        validate_agent(make_agent, GymEnvironment('LunarLanderContinuous-v2'))


if __name__ == '__main__':
    unittest.main()
