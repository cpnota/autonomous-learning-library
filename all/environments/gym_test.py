import unittest

import gym
import gymnasium
import torch

from all.environments import GymEnvironment


class GymEnvironmentTest(unittest.TestCase):
    def test_env_name(self):
        env = GymEnvironment("CartPole-v0")
        self.assertEqual(env.name, "CartPole-v0")

    def test_reset(self):
        env = GymEnvironment("CartPole-v0")
        state = env.reset()
        self.assertEqual(state.observation.shape, (4,))
        self.assertEqual(state.reward, 0)
        self.assertFalse(state.done)
        self.assertEqual(state.mask, 1)

    def test_step(self):
        env = GymEnvironment("CartPole-v0")
        env.reset()
        state = env.step(1)
        self.assertEqual(state.observation.shape, (4,))
        self.assertEqual(state.reward, 1.0)
        self.assertFalse(state.done)
        self.assertEqual(state.mask, 1)

    def test_step_until_done(self):
        env = GymEnvironment("CartPole-v0")
        env.reset()
        for _ in range(100):
            state = env.step(1)
            if state.done:
                break
        self.assertEqual(state.observation.shape, (4,))
        self.assertEqual(state.reward, 1.0)
        self.assertTrue(state.done)
        self.assertEqual(state.mask, 0)

    def test_duplicate_default_params(self):
        env = GymEnvironment("CartPole-v0")
        duplicates = env.duplicate(5)
        for duplicate in duplicates._envs:
            self.assertEqual(duplicate._id, "CartPole-v0")
            self.assertEqual(duplicate._name, "CartPole-v0")
            self.assertEqual(env._device, torch.device("cpu"))
            self.assertEqual(env._gym, gymnasium)

    def test_duplicate_custom_params(self):
        class MyWrapper:
            def __init__(self, env):
                self._env = env

        env = GymEnvironment(
            "CartPole-v0",
            legacy_gym=True,
            name="legacy_cartpole",
            device="my_device",
            wrap_env=MyWrapper,
        )
        duplicates = env.duplicate(5)
        for duplicate in duplicates._envs:
            self.assertEqual(duplicate._id, "CartPole-v0")
            self.assertEqual(duplicate._name, "legacy_cartpole")
            self.assertEqual(env._device, "my_device")
            self.assertEqual(env._gym, gym)
