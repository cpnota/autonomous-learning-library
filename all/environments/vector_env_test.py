import unittest
import gym
import torch
from all.environments import GymVectorEnvironment, GymEnvironment, DuplicateEnvironment


def make_vec_env(num_envs=3):
    env = gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v0')] * num_envs)
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
        env.seed(5)
        env.reset()
        for _ in range(100):
            state = env.step(torch.ones(num_envs, dtype=torch.int32))
            if state.done[0]:
                break
        else:
            self.assertTrue(False)
        self.assertEqual(state[0].observation.shape, (4,))
        self.assertEqual(state[0].reward, 1.)
        self.assertTrue(state[0].done)
        self.assertEqual(state[0].mask, 0)

    def test_same_as_duplicate(self):
        n_envs = 3
        torch.manual_seed(42)
        env1 = DuplicateEnvironment([GymEnvironment('CartPole-v0') for i in range(n_envs)])
        env2 = GymVectorEnvironment(make_vec_env(n_envs), "CartPole-v0")
        env1.seed(42)
        env2.seed(42)
        state1 = env1.reset()
        state2 = env2.reset()
        self.assertEqual(env1.name, env2.name)
        self.assertEqual(env1.action_space.n, env2.action_space.n)
        self.assertEqual(env1.observation_space.shape, env2.observation_space.shape)
        self.assertEqual(env1.num_envs, 3)
        self.assertEqual(env2.num_envs, 3)
        act_space = env1.action_space
        for i in range(2):
            self.assertTrue(torch.all(torch.eq(state1.observation, state2.observation)))
            self.assertTrue(torch.all(torch.eq(state1.reward, state2.reward)))
            self.assertTrue(torch.all(torch.eq(state1.done, state2.done)))
            self.assertTrue(torch.all(torch.eq(state1.mask, state2.mask)))
            actions = torch.tensor([act_space.sample() for i in range(n_envs)])
            state1 = env1.step(actions)
            state2 = env2.step(actions)


if __name__ == "__main__":
    unittest.main()
