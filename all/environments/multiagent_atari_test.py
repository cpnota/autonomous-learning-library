import unittest
import torch
from all.environments.multiagent_atari import MultiAgentAtariEnv


class MultiAgentAtariEnvTest(unittest.TestCase):
    def test_init(self):
        MultiAgentAtariEnv('pong_classic_v0')
        MultiAgentAtariEnv('mario_bros_v1')
        MultiAgentAtariEnv('entombed_cooperative_v0')

    def test_name(self):
        env = MultiAgentAtariEnv('pong_classic_v0')
        self.assertEqual(env.name, 'pong_classic_v0')

    def test_state_spaces(self):
        env = MultiAgentAtariEnv('pong_classic_v0')
        state_spaces = env.state_spaces
        self.assertEqual(state_spaces['first_0'].shape, (84, 84))
        self.assertEqual(state_spaces['second_0'].shape, (84, 84))

    def test_action_spaces(self):
        env = MultiAgentAtariEnv('pong_classic_v0')
        action_spaces = env.action_spaces
        print(action_spaces)

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
