import unittest
import torch
import torch_testing as tt
from all.core import State, StateArray
from all.bodies import TimeFeature


class TestAgent():
    def __init__(self):
        self.last_state = None

    def act(self, state):
        self.last_state = state
        return torch.zeros(len(state))


class TimeFeatureTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.test_agent = TestAgent()
        self.agent = TimeFeature(self.test_agent)

    def test_init(self):
        state = State(torch.randn(4))
        self.agent.act(state)
        tt.assert_allclose(self.test_agent.last_state.observation, torch.tensor(
            [0.3923, -0.2236, -0.3195, -1.2050, 0.0000]), atol=1e-04)

    def test_single_env(self):
        state = State(torch.randn(4))
        self.agent.act(state)
        tt.assert_allclose(self.test_agent.last_state.observation, torch.tensor(
            [0.3923, -0.2236, -0.3195, -1.2050, 0.]), atol=1e-04)
        self.agent.act(state)
        tt.assert_allclose(self.test_agent.last_state.observation, torch.tensor(
            [0.3923, -0.2236, -0.3195, -1.2050, 1e-3]), atol=1e-04)
        self.agent.act(state)
        tt.assert_allclose(self.test_agent.last_state.observation, torch.tensor(
            [0.3923, -0.2236, -0.3195, -1.2050, 2e-3]), atol=1e-04)

    def test_reset(self):
        state = State(torch.randn(4))
        self.agent.act(state)
        tt.assert_allclose(self.test_agent.last_state.observation, torch.tensor(
            [0.3923, -0.2236, -0.3195, -1.2050, 0.0000]), atol=1e-04)
        self.agent.act(state)
        tt.assert_allclose(self.test_agent.last_state.observation, torch.tensor(
            [0.3923, -0.2236, -0.3195, -1.2050, 1e-3]), atol=1e-04)
        self.agent.act(State(state.observation, done=True))
        tt.assert_allclose(self.test_agent.last_state.observation, torch.tensor(
            [0.3923, -0.2236, -0.3195, -1.2050, 2e-3]), atol=1e-04)
        self.agent.act(State(state.observation))
        tt.assert_allclose(self.test_agent.last_state.observation, torch.tensor(
            [0.3923, -0.2236, -0.3195, -1.2050, 0.0000]), atol=1e-04)
        self.agent.act(state)
        tt.assert_allclose(self.test_agent.last_state.observation, torch.tensor(
            [0.3923, -0.2236, -0.3195, -1.2050, 1e-3]), atol=1e-04)

    def test_multi_env(self):
        state = StateArray(torch.randn(2, 2), (2,))
        self.agent.act(state)
        tt.assert_allclose(self.test_agent.last_state.observation, torch.tensor(
            [[0.3923, -0.2236, 0.], [-0.3195, -1.2050, 0.]]), atol=1e-04)
        self.agent.act(state)
        tt.assert_allclose(self.test_agent.last_state.observation, torch.tensor(
            [[0.3923, -0.2236, 1e-3], [-0.3195, -1.2050, 1e-3]]), atol=1e-04)
        self.agent.act(StateArray(state.observation, (2,), done=torch.tensor([False, True])))
        tt.assert_allclose(self.test_agent.last_state.observation, torch.tensor(
            [[0.3923, -0.2236, 2e-3], [-0.3195, -1.2050, 2e-3]]), atol=1e-04)
        self.agent.act(state)
        tt.assert_allclose(self.test_agent.last_state.observation, torch.tensor(
            [[0.3923, -0.2236, 3e-3], [-0.3195, -1.2050, 0.]]), atol=1e-04)
        self.agent.act(state)
        tt.assert_allclose(self.test_agent.last_state.observation, torch.tensor(
            [[0.3923, -0.2236, 4e-3], [-0.3195, -1.2050, 1e-3]]), atol=1e-04)


if __name__ == '__main__':
    unittest.main()
