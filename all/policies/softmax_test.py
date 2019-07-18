import unittest
import torch
from torch import nn
import torch_testing as tt
from all.environments import State
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
        self.policy = SoftmaxPolicy(self.model, optimizer, ACTIONS)

    def test_run(self):
        state = State(torch.randn(1, STATE_DIM))
        action = self.policy(state)
        self.assertEqual(action.item(), 0)
        state = State(torch.randn(1, STATE_DIM))
        action = self.policy(state)
        self.assertEqual(action.item(), 2)
        self.policy.reinforce(torch.tensor([-1, 1000000]).float())
        action = self.policy(state)
        self.assertEqual(action.item(), 2)

    def test_multi_action(self):
        states = State(torch.randn(3, STATE_DIM))
        actions = self.policy(states)
        tt.assert_equal(actions, torch.tensor([2, 2, 0]))
        self.policy.reinforce(torch.tensor([[1, 2, 3]]).float())

    def test_multi_batch_reinforce(self):
        self.policy(State(torch.randn(2, STATE_DIM)))
        self.policy(State(torch.randn(2, STATE_DIM)))
        self.policy(State(torch.randn(2, STATE_DIM)))
        self.policy.reinforce(torch.tensor([1, 2, 3, 4]).float())
        self.policy.reinforce(torch.tensor([1, 2]).float())
        with self.assertRaises(Exception):
            self.policy.reinforce(torch.tensor([1, 2]).float())

    def test_list(self):
        torch.manual_seed(1)
        states = State(torch.randn(3, STATE_DIM), torch.tensor([1, 0, 1]))
        actions = self.policy(states)
        tt.assert_equal(actions, torch.tensor([1, 2, 1]))
        self.policy.reinforce(torch.tensor([[1, 2, 3]]).float())

    def test_action_prob(self):
        torch.manual_seed(1)
        states = State(torch.randn(3, STATE_DIM), torch.tensor([1, 0, 1]))
        with torch.no_grad():
            actions = self.policy(states)
        log_probs = self.policy(states, action=actions)
        tt.assert_almost_equal(log_probs, torch.tensor([-1.59, -1.099, -1.528]), decimal=3)

    def test_custom_loss(self):
        def loss(log_probs):
            return -log_probs.mean()

        states = State(torch.randn(3, STATE_DIM), torch.tensor([1, 1, 1]))
        actions = self.policy.eval(states)

        # notice the values increase with each successive reinforce
        log_probs = self.policy(states, actions)
        tt.assert_almost_equal(log_probs, torch.tensor([-0.84, -0.62, -0.757]), decimal=3)
        self.policy.reinforce(loss)
        log_probs = self.policy(states, actions)
        tt.assert_almost_equal(log_probs, torch.tensor([-0.811, -0.561, -0.701]), decimal=3)
        self.policy.reinforce(loss)
        log_probs = self.policy(states, actions)
        tt.assert_almost_equal(log_probs, torch.tensor([-0.785, -0.51, -0.651]), decimal=3)


if __name__ == '__main__':
    unittest.main()
