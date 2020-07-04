import unittest
import numpy as np
import torch
import torch_testing as tt
from all.core import State, StateList

class StateTest(unittest.TestCase):
    def test_constructor_defaults(self):
        observation = torch.randn(3, 4)
        state = State(observation)
        tt.assert_equal(state.observation, observation)
        self.assertEqual(state.mask, 1.)
        self.assertEqual(state.done, False)
        self.assertEqual(state.reward, 0.)
        self.assertEqual(state.shape, ())

    def test_from_dict(self):
        observation = torch.randn(3, 4)
        state = State({
            'observation': observation, 
            'done': True,
            'mask': 1,
            'reward': 5.
        })
        tt.assert_equal(state.observation, observation)
        self.assertEqual(state.done, True)
        self.assertEqual(state.mask, 1.)
        self.assertEqual(state.reward, 5.)

    def test_auto_mask_true(self):
        observation = torch.randn(3, 4)
        state = State({
            'observation': observation, 
            'done': True,
            'reward': 5.
        })
        self.assertEqual(state.mask, 0.)

    def test_auto_mask_false(self):
        observation = torch.randn(3, 4)
        state = State({
            'observation': observation, 
            'done': False,
            'reward': 5.
        })
        self.assertEqual(state.mask, 1.)

    def test_from_gym_reset(self):
        observation = np.array([1, 2, 3])
        state = State.from_gym(observation)
        tt.assert_equal(state.observation, torch.from_numpy(observation))
        self.assertEqual(state.mask, 1.)
        self.assertEqual(state.done, False)
        self.assertEqual(state.reward, 0.)
        self.assertEqual(state.shape, ())

    def test_from_gym_step(self):
        observation = np.array([1, 2, 3])
        state = State.from_gym((observation, 2., True, {'coolInfo': 3.}))
        tt.assert_equal(state.observation, torch.from_numpy(observation))
        self.assertEqual(state.mask, 0.)
        self.assertEqual(state.done, True)
        self.assertEqual(state.reward, 2.)
        self.assertEqual(state['coolInfo'], 3.)
        self.assertEqual(state.shape, ())

    def test_as_input(self):
        observation = torch.randn(3, 4)
        state = State(observation)
        self.assertEqual(state.as_input('observation').shape, (1, 3, 4))

    def test_as_output(self):
        observation = torch.randn(3, 4)
        state = State(observation)
        tensor = torch.randn(1, 5, 3)
        self.assertEqual(state.as_output(tensor).shape, (5, 3))

    def test_apply_mask(self):
        observation = torch.randn(3, 4)
        state = State.from_gym((observation, 0., True, {}))
        tt.assert_equal(state.apply_mask(observation), torch.zeros(3, 4))

    def test_apply(self):
        observation = torch.randn(3, 4)
        state = State(observation)
        model = torch.nn.Conv1d(3, 5, 2)
        output = state.apply(model, 'observation')
        self.assertEqual(output.shape, (5, 3))
        self.assertNotEqual(output.sum().item(), 0)


    def test_apply_done(self):
        observation = torch.randn(3, 4)
        state = State.from_gym((observation, 0., True, {}))
        model = torch.nn.Conv1d(3, 5, 2)
        output = state.apply(model, 'observation')
        self.assertEqual(output.shape, (5, 3))
        self.assertEqual(output.sum().item(), 0)

class StateListTest(unittest.TestCase):
    def test_constructor_defaults(self):
        raw = torch.randn(3, 4)
        state = State(raw, (3,))
        tt.assert_equal(state.observation, raw)
        self.assertEqual(state.mask, 1.)
        self.assertEqual(state.done, False)
        self.assertEqual(state.reward, 0.)

    def test_constructor_defaults(self):
        raw = torch.randn(3, 4)
        state = State(raw)
        tt.assert_equal(state.observation, raw)
        self.assertEqual(state.mask, 1.)
        self.assertEqual(state.done, False)
        self.assertEqual(state.reward, 0.)

    def test_custom_constructor_args(self):
        raw = torch.randn(3, 4)
        mask = torch.zeros(3)
        info = ['a', 'b', 'c']
        state = State(raw, mask=mask, info=info)
        tt.assert_equal(state.features, raw)
        tt.assert_equal(state.mask, torch.zeros(3))
        self.assertEqual(state.info, info)

    def test_not_done(self):
        state = State(torch.randn(1, 4))
        self.assertFalse(state.done)

    def test_done(self):
        raw = torch.randn(1, 4)
        state = State(raw, mask=DONE)
        self.assertTrue(state.done)

    def test_from_list(self):
        state1 = State(torch.randn(1, 4), mask=DONE, info=['a'])
        state2 = State(torch.randn(1, 4), mask=NOT_DONE, info=['b'])
        state3 = State(torch.randn(1, 4))
        state = State.from_list([state1, state2, state3])
        tt.assert_equal(state.raw, torch.cat((state1.raw, state2.raw, state3.raw)))
        tt.assert_equal(state.mask, torch.tensor([0, 1, 1]))
        self.assertEqual(state.info, ['a', 'b', None])

    def test_from_gym(self):
        gym_obs = np.array([1, 2, 3])
        done = True
        info = 'a'
        state = State.from_gym(gym_obs, done, info)
        tt.assert_equal(state.raw, torch.tensor([[1, 2, 3]]))
        tt.assert_equal(state.mask, DONE)
        self.assertEqual(state.info, ['a'])

    def test_get_item(self):
        raw = torch.randn(3, 4)
        states = State(raw)
        state = states[2]
        tt.assert_equal(state.raw, raw[2].unsqueeze(0))
        tt.assert_equal(state.mask, NOT_DONE)
        self.assertEqual(state.info, [None])

    def test_len(self):
        state = State(torch.randn(3, 4))
        self.assertEqual(len(state), 3)
