import unittest
from all.environments import MujocoEnvironment, GymEnvironment


class MujocoEnvironmentTest(unittest.TestCase):
    def test_load_env(self):
        env = MujocoEnvironment("Ant-v4")
        self.assertEqual(env.name, 'Ant-v4')

    def test_observation_space(self):
        env = MujocoEnvironment("Ant-v4")
        self.assertEqual(env.observation_space.shape, (27,))

    def test_action_space(self):
        env = MujocoEnvironment("Ant-v4")
        self.assertEqual(env.action_space.shape, (8,))

    def test_reset(self):
        env = MujocoEnvironment("Ant-v4")
        state = env.reset(seed=0)
        self.assertEqual(state.observation.shape, (27,))
        self.assertEqual(state.reward, 0.)
        self.assertFalse(state.done)
        self.assertEqual(state.mask, 1)

    def test_step(self):
        env = MujocoEnvironment("Ant-v4")
        state = env.reset(seed=0)
        state = env.step(env.action_space.sample())
        self.assertEqual(state.observation.shape, (27,))
        self.assertGreater(state.reward, -2.)
        self.assertLess(state.reward, 2)
        self.assertNotEqual(state.reward, 0.)
        self.assertFalse(state.done)
        self.assertEqual(state.mask, 1)
