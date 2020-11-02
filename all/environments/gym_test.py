import unittest
import gym
from all.environments import GymEnvironment


class GymEnvironmentTest(unittest.TestCase):
    def test_env_name(self):
        env = GymEnvironment('CartPole-v0')
        self.assertEqual(env.name, 'CartPole-v0')

    def test_preconstructed_env_name(self):
        env = GymEnvironment(gym.make('Blackjack-v0'))
        self.assertEqual(env.name, 'BlackjackEnv')

    def test_reset(self):
        env = GymEnvironment('CartPole-v0')
        state = env.reset()
        self.assertEqual(state.observation.shape, (4,))
        self.assertEqual(state.reward, 0)
        self.assertFalse(state.done)
        self.assertEqual(state.mask, 1)

    def test_reset_preconstructed_env(self):
        env = GymEnvironment(gym.make('CartPole-v0'))
        state = env.reset()
        self.assertEqual(state.observation.shape, (4,))
        self.assertEqual(state.reward, 0)
        self.assertFalse(state.done)
        self.assertEqual(state.mask, 1)

    def test_step(self):
        env = GymEnvironment('CartPole-v0')
        env.reset()
        state = env.step(1)
        self.assertEqual(state.observation.shape, (4,))
        self.assertEqual(state.reward, 1.)
        self.assertFalse(state.done)
        self.assertEqual(state.mask, 1)

    def test_step_until_done(self):
        env = GymEnvironment('CartPole-v0')
        env.reset()
        for _ in range(100):
            state = env.step(1)
            if state.done:
                break
        self.assertEqual(state.observation.shape, (4,))
        self.assertEqual(state.reward, 1.)
        self.assertTrue(state.done)
        self.assertEqual(state.mask, 0)
