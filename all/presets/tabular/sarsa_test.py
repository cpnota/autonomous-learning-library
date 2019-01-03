import unittest
from all.environments import GymWrapper
from all.presets.tabular import sarsa


class TestTabularSarsa(unittest.TestCase):
    def test_runs(self):
        env = GymWrapper('FrozenLake-v0')
        agent = sarsa(env)

        env.reset()
        agent.new_episode(env)

        for i in range(0, 3):
            agent.act()
            if env.done:
                env.reset()
                agent.new_episode(env)


if __name__ == '__main__':
    unittest.main()
