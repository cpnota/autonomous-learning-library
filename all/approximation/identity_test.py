import unittest
import torch
import torch_testing as tt
from all.core import State
from all.approximation import Identity, FixedTarget


class TestIdentityNetwork(unittest.TestCase):
    def setUp(self):
        self.model = Identity('cpu', target=FixedTarget(10))

    def test_forward_tensor(self):
        inputs = torch.tensor([1, 2, 3])
        outputs = self.model(inputs)
        tt.assert_equal(inputs, outputs)

    def test_forward_state(self):
        inputs = State({
            'observation': torch.tensor([1, 2, 3])
        })
        outputs = self.model(inputs)
        self.assertEqual(inputs, outputs)

    def test_eval(self):
        inputs = torch.tensor([1, 2, 3])
        outputs = self.model.target(inputs)
        tt.assert_equal(inputs, outputs)

    def test_target(self):
        inputs = torch.tensor([1, 2, 3])
        outputs = self.model.target(inputs)
        tt.assert_equal(inputs, outputs)

    def test_reinforce(self):
        self.model.reinforce()

    def test_step(self):
        self.model.step()


if __name__ == "__main__":
    unittest.main()
