import unittest
from all.environments import GymWrapper
from all.presets.fourier import actor_critic


class TestActorCritic(unittest.TestCase):
    def test_runs(self):
        env = GymWrapper('MountainCar-v0')
        agent = actor_critic(env)

        env.reset()
        agent.new_episode(env)

        agent.act()
        agent.act()
        agent.act()

        self.assertIsNotNone(env.state)


if __name__ == '__main__':
    unittest.main()
