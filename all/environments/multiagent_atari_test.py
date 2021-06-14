import unittest
import torch
from all.environments import MultiagentAtariEnv


class MultiagentAtariEnvTest(unittest.TestCase):
    def test_init(self):
        MultiagentAtariEnv('pong_v2', device='cpu')
        MultiagentAtariEnv('mario_bros_v2', device='cpu')
        MultiagentAtariEnv('entombed_cooperative_v2', device='cpu')

    def test_reset(self):
        env = MultiagentAtariEnv('pong_v2', device='cpu')
        state = env.reset()
        self.assertEqual(state.observation.shape, (1, 84, 84))
        self.assertEqual(state.reward, 0)
        self.assertEqual(state.done, False)
        self.assertEqual(state.mask, 1.)
        self.assertEqual(state['agent'], 'first_0')

    def test_step(self):
        env = MultiagentAtariEnv('pong_v2', device='cpu')
        env.reset()
        state = env.step(0)
        self.assertEqual(state.observation.shape, (1, 84, 84))
        self.assertEqual(state.reward, 0)
        self.assertEqual(state.done, False)
        self.assertEqual(state.mask, 1.)
        self.assertEqual(state['agent'], 'second_0')

    def test_step_tensor(self):
        env = MultiagentAtariEnv('pong_v2', device='cpu')
        env.reset()
        state = env.step(torch.tensor([0]))
        self.assertEqual(state.observation.shape, (1, 84, 84))
        self.assertEqual(state.reward, 0)
        self.assertEqual(state.done, False)
        self.assertEqual(state.mask, 1.)
        self.assertEqual(state['agent'], 'second_0')

    def test_name(self):
        env = MultiagentAtariEnv('pong_v2', device='cpu')
        self.assertEqual(env.name, 'pong_v2')

    def test_agent_iter(self):
        env = MultiagentAtariEnv('pong_v2', device='cpu')
        env.reset()
        it = iter(env.agent_iter())
        self.assertEqual(next(it), 'first_0')

    def test_state_spaces(self):
        state_spaces = MultiagentAtariEnv('pong_v2', device='cpu').state_spaces
        self.assertEqual(state_spaces['first_0'].shape, (1, 84, 84))
        self.assertEqual(state_spaces['second_0'].shape, (1, 84, 84))

    def test_action_spaces(self):
        action_spaces = MultiagentAtariEnv('pong_v2', device='cpu').action_spaces
        self.assertEqual(action_spaces['first_0'].n, 18)
        self.assertEqual(action_spaces['second_0'].n, 18)

    def test_list_agents(self):
        env = MultiagentAtariEnv('pong_v2', device='cpu')
        self.assertEqual(env.agents, ['first_0', 'second_0'])

    def test_is_done(self):
        env = MultiagentAtariEnv('pong_v2', device='cpu')
        env.reset()
        self.assertFalse(env.is_done('first_0'))
        self.assertFalse(env.is_done('second_0'))

    def test_last(self):
        env = MultiagentAtariEnv('pong_v2', device='cpu')
        env.reset()
        state = env.last()
        self.assertEqual(state.observation.shape, (1, 84, 84))
        self.assertEqual(state.reward, 0)
        self.assertEqual(state.done, False)
        self.assertEqual(state.mask, 1.)
        self.assertEqual(state['agent'], 'first_0')


if __name__ == "__main__":
    unittest.main()
