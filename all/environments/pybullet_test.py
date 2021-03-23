import unittest
from all.environments import PybulletEnvironment, GymEnvironment


class PybulletEnvironmentTest(unittest.TestCase):
    def test_env_short_name(self):
        for short_name, long_name in PybulletEnvironment.short_names.items():
            env = PybulletEnvironment(short_name)
            self.assertEqual(env.name, long_name)

    def test_env_full_name(self):
        env = PybulletEnvironment('HalfCheetahBulletEnv-v0')
        self.assertEqual(env.name, 'HalfCheetahBulletEnv-v0')

    def test_reset(self):
        env = PybulletEnvironment('cheetah')
        state = env.reset()
        self.assertEqual(state.observation.shape, (26,))
        self.assertEqual(state.reward, 0.)
        self.assertFalse(state.done)
        self.assertEqual(state.mask, 1)

    def test_step(self):
        env = PybulletEnvironment('cheetah')
        env.seed(0)
        state = env.reset()
        state = env.step(env.action_space.sample())
        self.assertEqual(state.observation.shape, (26,))
        self.assertGreater(state.reward, -1.)
        self.assertLess(state.reward, 1)
        self.assertNotEqual(state.reward, 0.)
        self.assertFalse(state.done)
        self.assertEqual(state.mask, 1)
