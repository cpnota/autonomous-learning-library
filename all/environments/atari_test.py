import unittest
from all.environments import AtariEnvironment


class AtariEnvironmentTest(unittest.TestCase):
    def test_reset(self):
        env = AtariEnvironment('Breakout')
        state = env.reset()
        self.assertEqual(state.observation.shape, (1, 84, 84))
        self.assertEqual(state.reward, 0)
        self.assertFalse(state.done)
        self.assertEqual(state.mask, 1)

    def test_step(self):
        env = AtariEnvironment('Breakout')
        env.reset()
        state = env.step(1)
        self.assertEqual(state.observation.shape, (1, 84, 84))
        self.assertEqual(state.reward, 0)
        self.assertFalse(state.done)
        self.assertEqual(state.mask, 1)
        self.assertEqual(state['life_lost'], False)

    def test_step_until_life_lost(self):
        env = AtariEnvironment('Breakout')
        env.reset()
        for _ in range(100):
            state = env.step(1)
            if state['life_lost']:
                break
        self.assertEqual(state.observation.shape, (1, 84, 84))
        self.assertEqual(state.reward, 0)
        self.assertFalse(state.done)
        self.assertEqual(state.mask, 1)
        self.assertEqual(state['life_lost'], True)

    def test_step_until_done(self):
        env = AtariEnvironment('Breakout')
        env.reset()
        for _ in range(1000):
            state = env.step(1)
            if state.done:
                break
        self.assertEqual(state.observation.shape, (1, 84, 84))
        self.assertEqual(state.reward, 0)
        self.assertTrue(state.done)
        self.assertEqual(state.mask, 0)
        self.assertEqual(state['life_lost'], False)
