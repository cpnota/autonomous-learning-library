import unittest
import warnings
import numpy as np
import torch
import torch_testing as tt
from all.core import State, StateArray


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

    def test_to_device(self):
        observation = torch.randn(3, 4)
        state = State(observation, device=torch.device('cpu'))
        state_cpu = state.to("cpu")
        self.assertTrue(torch.equal(state['observation'], state_cpu['observation']))
        self.assertFalse(state is state_cpu)


class StateArrayTest(unittest.TestCase):
    def test_constructor_defaults(self):
        raw = torch.randn(3, 4)
        state = State(raw, (3,))
        tt.assert_equal(state.observation, raw)
        self.assertEqual(state.mask, 1.)
        self.assertEqual(state.done, False)
        self.assertEqual(state.reward, 0.)

    def test_apply(self):
        observation = torch.randn(3, 4)
        state = StateArray(observation, (3,))
        model = torch.nn.Linear(4, 2)
        output = state.apply(model, 'observation')
        self.assertEqual(output.shape, (3, 2))
        self.assertNotEqual(output.sum().item(), 0)

    def test_apply_done(self):
        observation = torch.randn(3, 4)
        state = StateArray(observation, (3,), mask=torch.tensor([0., 0., 0.]))
        model = torch.nn.Linear(4, 2)
        output = state.apply(model, 'observation')
        self.assertEqual(output.shape, (3, 2))
        self.assertEqual(output.sum().item(), 0)

    def test_as_output(self):
        observation = torch.randn(3, 4)
        state = StateArray(observation, (3,))
        tensor = torch.randn(3, 5)
        self.assertEqual(state.as_output(tensor).shape, (3, 5))

    def test_auto_mask(self):
        observation = torch.randn(3, 4)
        state = StateArray({
            'observation': observation,
            'done': torch.tensor([True, False, True]),
        }, (3,))
        tt.assert_equal(state.mask, torch.tensor([0., 1., 0.]))

    def test_multi_dim(self):
        state = StateArray.array([
            State(torch.randn((3, 4))),
            State(torch.randn((3, 4)))
        ])
        self.assertEqual(state.shape, (2,))
        state = StateArray.array([state] * 3)
        self.assertEqual(state.shape, (3, 2))
        state = StateArray.array([state] * 5)
        self.assertEqual(state.shape, (5, 3, 2))
        tt.assert_equal(state.mask, torch.ones((5, 3, 2)))
        tt.assert_equal(state.done, torch.zeros((5, 3, 2)).bool())
        tt.assert_equal(state.reward, torch.zeros((5, 3, 2)))

    def test_view(self):
        state = StateArray.array([
            State(torch.randn((3, 4))),
            State(torch.randn((3, 4)))
        ])
        self.assertEqual(state.shape, (2,))
        state = StateArray.array([state] * 3)
        self.assertEqual(state.shape, (3, 2))
        state = state.view((2, 3))
        self.assertEqual(state.shape, (2, 3))
        self.assertEqual(state.observation.shape, (2, 3, 3, 4))

    def test_key_error(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            StateArray.array([
                State({
                    'observation': torch.tensor([1, 2]),
                    'other_key': True
                }),
                State({
                    'observation': torch.tensor([1, 2]),
                }),
            ])
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].message.args[0], 'KeyError while creating StateArray for key "other_key", omitting.')

    def test_type_error(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            StateArray.array([
                State({
                    'observation': torch.tensor([1, 2]),
                    'other_key': torch.tensor([1])
                }),
                State({
                    'observation': torch.tensor([1, 2]),
                    'other_key': 5.
                }),
            ])
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].message.args[0], 'TypeError while creating StateArray for key "other_key", omitting.')


if __name__ == "__main__":
    unittest.main()
