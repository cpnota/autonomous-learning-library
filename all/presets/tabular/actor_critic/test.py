import unittest
from all.environments import GymWrapper
from all.presets.tabular import actor_critic


class TestSarsa(unittest.TestCase):
    def test_runs(self):
        env = GymWrapper('FrozenLake-v0')
        agent = actor_critic(env)

        env.reset()
        agent.new_episode(env)

        while not env.done:
            agent.act()


if __name__ == '__main__':
    unittest.main()
