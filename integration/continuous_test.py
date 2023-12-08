import unittest
from all.environments import GymEnvironment, PybulletEnvironment, MujocoEnvironment
from all.presets.continuous import ddpg, ppo, sac
from validate_agent import validate_agent


class TestContinuousPresets(unittest.TestCase):
    def test_ddpg(self):
        validate_agent(
            ddpg.device("cpu").hyperparameters(replay_start_size=50),
            GymEnvironment("LunarLanderContinuous-v2"),
        )

    def test_ppo(self):
        validate_agent(ppo.device("cpu"), GymEnvironment("LunarLanderContinuous-v2"))

    def test_sac(self):
        validate_agent(
            sac.device("cpu").hyperparameters(replay_start_size=50),
            GymEnvironment("LunarLanderContinuous-v2"),
        )

    def test_mujoco(self):
        validate_agent(
            sac.device("cpu").hyperparameters(replay_start_size=50),
            MujocoEnvironment("HalfCheetah-v4"),
        )

    def test_pybullet(self):
        validate_agent(
            sac.device("cpu").hyperparameters(replay_start_size=50),
            PybulletEnvironment("cheetah"),
        )


if __name__ == "__main__":
    unittest.main()
