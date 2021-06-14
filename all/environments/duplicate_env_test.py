import unittest
import gym
import torch
from all.environments import DuplicateEnvironment, GymEnvironment


def make_vec_env(num_envs=3):
    env = [GymEnvironment('CartPole-v0') for i in range(num_envs)]
    return env


class DuplicateEnvironmentTest(unittest.TestCase):
    def test_env_name(self):
        env = DuplicateEnvironment(make_vec_env())
        self.assertEqual(env.name, 'CartPole-v0')

    def test_num_envs(self):
        num_envs = 5
        env = DuplicateEnvironment(make_vec_env(num_envs))
        self.assertEqual(env.num_envs, num_envs)
        self.assertEqual((num_envs,), env.reset().shape)

    def test_reset(self):
        num_envs = 5
        env = DuplicateEnvironment(make_vec_env(num_envs))
        state = env.reset()
        self.assertEqual(state.observation.shape, (num_envs, 4))
        self.assertTrue((state.reward == torch.zeros(num_envs, )).all())
        self.assertTrue((state.done == torch.zeros(num_envs, )).all())
        self.assertTrue((state.mask == torch.ones(num_envs, )).all())

    def test_step(self):
        num_envs = 5
        env = DuplicateEnvironment(make_vec_env(num_envs))
        env.reset()
        state = env.step(torch.ones(num_envs, dtype=torch.int32))
        self.assertEqual(state.observation.shape, (num_envs, 4))
        self.assertTrue((state.reward == torch.ones(num_envs, )).all())
        self.assertTrue((state.done == torch.zeros(num_envs, )).all())
        self.assertTrue((state.mask == torch.ones(num_envs, )).all())

    def test_step_until_done(self):
        num_envs = 3
        env = DuplicateEnvironment(make_vec_env(num_envs))
        env.seed(5)
        env.reset()
        for _ in range(100):
            state = env.step(torch.ones(num_envs, dtype=torch.int32))
            if state.done[0]:
                break
        self.assertEqual(state[0].observation.shape, (4,))
        self.assertEqual(state[0].reward, 1.)
        self.assertTrue(state[0].done)
        self.assertEqual(state[0].mask, 0)
