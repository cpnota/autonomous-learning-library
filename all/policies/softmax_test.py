import unittest
import torch
from torch import nn
import torch_testing as tt
from all.core import State
from all.policies import SoftmaxPolicy

STATE_DIM = 2
ACTIONS = 3


class TestSoftmax(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, ACTIONS)
        )
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.policy = SoftmaxPolicy(self.model, optimizer)

    def test_run(self):
        state1 = State(torch.randn(1, STATE_DIM))
        dist1 = self.policy(state1)
        action1 = dist1.sample()
        log_prob1 = dist1.log_prob(action1)
        self.assertEqual(action1.item(), 0)

        state2 = State(torch.randn(1, STATE_DIM))
        dist2 = self.policy(state2)
        action2 = dist2.sample()
        log_prob2 = dist2.log_prob(action2)
        self.assertEqual(action2.item(), 2)

        loss = -(torch.tensor([-1, 1000000]) * torch.cat((log_prob1, log_prob2))).mean()
        self.policy.reinforce(loss)

        state3 = State(torch.randn(1, STATE_DIM))
        dist3 = self.policy(state3)
        action3 = dist3.sample()
        self.assertEqual(action3.item(), 2)

    def test_multi_action(self):
        states = State(torch.randn(3, STATE_DIM))
        actions = self.policy(states).sample()
        tt.assert_equal(actions, torch.tensor([2, 2, 0]))

    def test_list(self):
        torch.manual_seed(1)
        states = State(torch.randn(3, STATE_DIM), torch.tensor([1, 0, 1]))
        dist = self.policy(states)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        tt.assert_equal(actions, torch.tensor([1, 2, 1]))
        loss = -(torch.tensor([[1, 2, 3]]) * log_probs).mean()
        self.policy.reinforce(loss)

    def test_reinforce(self):
        def loss(log_probs):
            return -log_probs.mean()

        states = State(torch.randn(3, STATE_DIM), torch.tensor([1, 1, 1]))
        actions = self.policy.no_grad(states).sample()

        # notice the values increase with each successive reinforce
        log_probs = self.policy(states).log_prob(actions)
        tt.assert_almost_equal(log_probs, torch.tensor([-0.84, -0.62, -0.757]), decimal=3)
        self.policy.reinforce(loss(log_probs))
        log_probs = self.policy(states).log_prob(actions)
        tt.assert_almost_equal(log_probs, torch.tensor([-0.811, -0.561, -0.701]), decimal=3)
        self.policy.reinforce(loss(log_probs))
        log_probs = self.policy(states).log_prob(actions)
        tt.assert_almost_equal(log_probs, torch.tensor([-0.785, -0.51, -0.651]), decimal=3)

    def test_eval(self):
        states = State(torch.randn(3, STATE_DIM), torch.tensor([1, 1, 1]))
        dist = self.policy.no_grad(states)
        tt.assert_almost_equal(dist.probs, torch.tensor([
            [0.352, 0.216, 0.432],
            [0.266, 0.196, 0.538],
            [0.469, 0.227, 0.304]
        ]), decimal=3)
        best = self.policy.eval(states).sample()
        tt.assert_equal(best, torch.tensor([2, 2, 0]))


if __name__ == '__main__':
    unittest.main()
