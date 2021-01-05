import unittest
import torch
from all.environments import MultiAgentAtariEnv


class MultiAgentAtariEnvTest(unittest.TestCase):
    def test_init(self):
        MultiAgentAtariEnv('pong_v1')
        MultiAgentAtariEnv('mario_bros_v2')
        MultiAgentAtariEnv('entombed_cooperative_v2')

    def test_reset(self):
        env = MultiAgentAtariEnv('pong_v1')
        state = env.reset()
        self.assertEqual(state.observation.shape, (1, 84, 84))
        self.assertEqual(state.reward, 0)
        self.assertEqual(state.done, False)
        self.assertEqual(state.mask, 1.)
        self.assertEqual(state['agent'], 'first_0')

    def test_step(self):
        env = MultiAgentAtariEnv('pong_v1')
        env.reset()
        state = env.step(0)
        self.assertEqual(state.observation.shape, (1, 84, 84))
        self.assertEqual(state.reward, 0)
        self.assertEqual(state.done, False)
        self.assertEqual(state.mask, 1.)
        self.assertEqual(state['agent'], 'second_0')

    def test_step_tensor(self):
        env = MultiAgentAtariEnv('pong_v1')
        env.reset()
        state = env.step(torch.tensor([0]))
        self.assertEqual(state.observation.shape, (1, 84, 84))
        self.assertEqual(state.reward, 0)
        self.assertEqual(state.done, False)
        self.assertEqual(state.mask, 1.)
        self.assertEqual(state['agent'], 'second_0')

    def test_name(self):
        env = MultiAgentAtariEnv('pong_v1')
        self.assertEqual(env.name, 'pong_v1')

    def test_agent_iter(self):
        env = MultiAgentAtariEnv('pong_v1')
        env.reset()
        it = iter(env.agent_iter())
        self.assertEqual(next(it), 'first_0')

    def test_state_spaces(self):
        state_spaces = MultiAgentAtariEnv('pong_v1').state_spaces
        self.assertEqual(state_spaces['first_0'].shape, (1, 84, 84))
        self.assertEqual(state_spaces['second_0'].shape, (1, 84, 84))

    def test_action_spaces(self):
        action_spaces = MultiAgentAtariEnv('pong_v1').action_spaces
        self.assertEqual(action_spaces['first_0'].n, 18)
        self.assertEqual(action_spaces['second_0'].n, 18)

    def test_list_agents(self):
        env = MultiAgentAtariEnv('pong_v1')
        self.assertEqual(env.agents, ['first_0', 'second_0'])

    def test_is_done(self):
        env = MultiAgentAtariEnv('pong_v1')
        env.reset()
        self.assertFalse(env.is_done('first_0'))
        self.assertFalse(env.is_done('second_0'))

    def test_last(self):
        env = MultiAgentAtariEnv('pong_v1')
        env.reset()
        state = env.last()
        self.assertEqual(state.observation.shape, (1, 84, 84))
        self.assertEqual(state.reward, 0)
        self.assertEqual(state.done, False)
        self.assertEqual(state.mask, 1.)
        self.assertEqual(state['agent'], 'first_0')

if __name__ == "__main__":
    unittest.main()
