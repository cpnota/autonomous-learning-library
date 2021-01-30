import unittest
import gym
import torch
from all.environments import GymVectorEnvironment


def make_vec_env(num_envs=3):
    env = gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v0')]*num_envs)
    return env

class GymVectorEnvironmentTest(unittest.TestCase):
    def test_env_name(self):
        env = GymVectorEnvironment(make_vec_env(), "CartPole")
        self.assertEqual(env.name, 'CartPole')

    def test_num_envs(self):
        num_envs = 5
        env = GymVectorEnvironment(make_vec_env(num_envs), "CartPole")
        self.assertEqual(env.num_envs, num_envs)
        self.assertEqual((num_envs,), env.reset().shape)

    def test_reset(self):
        num_envs = 5
        env = GymVectorEnvironment(make_vec_env(num_envs), "CartPole")
        state = env.reset()
        self.assertEqual(state.observation.shape, (num_envs, 4))
        self.assertTrue((state.reward == torch.zeros(num_envs, )).all())
        self.assertTrue((state.done == torch.zeros(num_envs, )).all())
        self.assertTrue((state.mask == torch.ones(num_envs, )).all())

    def test_step(self):
        num_envs = 5
        env = GymVectorEnvironment(make_vec_env(num_envs), "CartPole")
        env.reset()
        state = env.step(torch.ones(num_envs, dtype=torch.int32))
        self.assertEqual(state.observation.shape, (num_envs, 4))
        self.assertTrue((state.reward == torch.ones(num_envs, )).all())
        self.assertTrue((state.done == torch.zeros(num_envs, )).all())
        self.assertTrue((state.mask == torch.ones(num_envs, )).all())

    def test_step_until_done(self):
        num_envs = 3
        env = GymVectorEnvironment(make_vec_env(num_envs), "CartPole")
        env.reset()
        for _ in range(100):
            state = env.step(torch.ones(num_envs, dtype=torch.int32))
            if state.done[0]:
                break
        self.assertEqual(state[0].observation.shape, (4,))
        self.assertEqual(state[0].reward, 1.)
        self.assertTrue(state[0].done)
        self.assertEqual(state[0].mask, 0)
