import unittest
import torch
from all.environments.multiagent_atari import MultiAgentAtariEnv


class MultiAgentAtariEnvTest(unittest.TestCase):
    def test_init(self):
        MultiAgentAtariEnv('pong_classic_v0')
        MultiAgentAtariEnv('mario_bros_v1')
        MultiAgentAtariEnv('entombed_cooperative_v0')

    def test_list_agents(self):
        env = MultiAgentAtariEnv('pong_classic_v0')
        self.assertEquals(env.agents, ['first_0', 'second_0'])

    def test_reset(self):
        env = MultiAgentAtariEnv('pong_classic_v0')
        state = env.reset()
        self.assertEqual(state.observation.shape, (1, 84, 84))
        self.assertEqual(state.reward, 0)
        self.assertEqual(state.done, False)
        self.assertEqual(state.mask, 1.)

    def test_step(self):
        env = MultiAgentAtariEnv('pong_classic_v0')
        state = env.reset()
        env.step(torch.tensor(1))
        self.assertEqual(state.observation.shape, (1, 84, 84))
        self.assertEqual(state.reward, 0)
        self.assertEqual(state.done, False)
        self.assertEqual(state.mask, 1.)


if __name__ == "__main__":
    unittest.main()
