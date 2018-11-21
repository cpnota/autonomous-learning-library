import unittest
from all.environments import GymWrapper
from all.presets.tabular import sarsa


class TestTabularSarsa(unittest.TestCase):
    def test_runs(self):
        env = GymWrapper('FrozenLake-v0')
        agent = sarsa(env)

        env.reset()
        agent.new_episode(env)

        agent.act()
        agent.act()
        agent.act()

        self.assertIsNotNone(env.state)


if __name__ == '__main__':
    unittest.main()
