import unittest
import torch
from torch import nn
import torch_testing as tt
from all.environments import State
from all.approximation import QDist

STATE_DIM = 1
ACTIONS = 2
ATOMS = 5
V_MIN = -2
V_MAX = 2


class TestQDist(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(nn.Linear(STATE_DIM, ACTIONS * ATOMS))
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.q = QDist(self.model, optimizer, ACTIONS, ATOMS, V_MIN, V_MAX)

    def test_atoms(self):
        tt.assert_almost_equal(self.q.atoms, torch.tensor([-2, -1, 0, 1, 2]))

    def test_q_values(self):
        states = State(torch.randn((3, STATE_DIM)))
        probs = self.q(states)
        self.assertEqual(probs.shape, (3, ACTIONS, ATOMS))
        tt.assert_almost_equal(
            probs.sum(dim=2),
            torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            decimal=3,
        )
        tt.assert_almost_equal(
            probs,
            torch.tensor(
                [
                    [
                        [0.2065, 0.1045, 0.1542, 0.2834, 0.2513],
                        [0.3903, 0.2471, 0.0360, 0.1733, 0.1533],
                    ],
                    [
                        [0.1966, 0.1299, 0.1431, 0.3167, 0.2137],
                        [0.3190, 0.2471, 0.0534, 0.1424, 0.2380],
                    ],
                    [
                        [0.1427, 0.2486, 0.0946, 0.4112, 0.1029],
                        [0.0819, 0.1320, 0.1203, 0.0373, 0.6285],
                    ],
                ]
            ),
            decimal=3,
        )

    def test_single_q_values(self):
        states = State(torch.randn((3, STATE_DIM)))
        actions = torch.tensor([0, 1, 0])
        probs = self.q(states, actions)
        self.assertEqual(probs.shape, (3, ATOMS))
        tt.assert_almost_equal(
            probs.sum(dim=1), torch.tensor([1.0, 1.0, 1.0]), decimal=3
        )
        tt.assert_almost_equal(
            probs,
            torch.tensor(
                [
                    [0.2065, 0.1045, 0.1542, 0.2834, 0.2513],
                    [0.3190, 0.2471, 0.0534, 0.1424, 0.2380],
                    [0.1427, 0.2486, 0.0946, 0.4112, 0.1029],
                ]
            ),
            decimal=3,
        )

    def test_done(self):
        states = State(torch.randn((3, STATE_DIM)), mask=torch.tensor([1, 0, 1]))
        probs = self.q(states)
        self.assertEqual(probs.shape, (3, ACTIONS, ATOMS))
        tt.assert_almost_equal(
            probs.sum(dim=2),
            torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            decimal=3,
        )
        tt.assert_almost_equal(
            probs,
            torch.tensor(
                [
                    [
                        [0.2065, 0.1045, 0.1542, 0.2834, 0.2513],
                        [0.3903, 0.2471, 0.0360, 0.1733, 0.1533],
                    ],
                    [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]],
                    [
                        [0.1427, 0.2486, 0.0946, 0.4112, 0.1029],
                        [0.0819, 0.1320, 0.1203, 0.0373, 0.6285],
                    ],
                ]
            ),
            decimal=3,
        )

    def test_reinforce(self):
        states = State(torch.randn((3, STATE_DIM)))
        actions = torch.tensor([0, 1, 0])
        original_probs = self.q(states, actions)
        tt.assert_almost_equal(
            original_probs,
            torch.tensor(
                [
                    [0.2065, 0.1045, 0.1542, 0.2834, 0.2513],
                    [0.3190, 0.2471, 0.0534, 0.1424, 0.2380],
                    [0.1427, 0.2486, 0.0946, 0.4112, 0.1029],
                ]
            ),
            decimal=3,
        )

        target_dists = torch.tensor(
            [[0, 0, 1, 0, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0]]
        ).float()
        self.q.reinforce(target_dists)

        new_probs = self.q(states, actions)
        tt.assert_almost_equal(
            torch.sign(new_probs - original_probs), torch.sign(target_dists - 0.5)
        )


if __name__ == "__main__":
    unittest.main()
