import unittest
from all.environments import GymWrapper
from all.presets.tabular import sarsa


class TestSarsa(unittest.TestCase):
    def test_runs(self):
        env = GymWrapper('FrozenLake-v0')
        agent = sarsa(env)

        env.reset()
        agent.new_episode(env)

        while not env.done:
            agent.act()


if __name__ == '__main__':
    unittest.main()
