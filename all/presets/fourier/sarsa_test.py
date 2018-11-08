from all.environments import GymWrapper
from all.presets.fourier import Sarsa
import unittest

class TestSarsa(unittest.TestCase):
  def testRuns(self):
    env = GymWrapper('MountainCar-v0')
    agent = Sarsa(env)

    env.reset()
    agent.new_episode(env)

    agent.act()
    agent.act()
    agent.act()

if __name__ == '__main__':
    unittest.main()
