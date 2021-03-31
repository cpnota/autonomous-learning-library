from torch.nn.functional import mse_loss
from ._parallel_agent import ParallelAgent
from .vqn import VQNTestAgent


class VSarsa(ParallelAgent):
    '''
    Vanilla SARSA (VSarsa).
    SARSA (State-Action-Reward-State-Action) is an on-policy alternative to Q-learning. Unlike Q-learning,
    SARSA attempts to learn the Q-function for the current policy rather than the optimal policy. This
    approach is more stable but may not result in the optimal policy. However, this problem can be mitigated
    by decaying the exploration rate over time.

    Args:
        q (QNetwork): An Approximation of the Q function.
        policy (GreedyPolicy): A policy derived from the Q-function.
        discount_factor (float): Discount factor for future rewards.
    '''

    def __init__(self, q, policy, discount_factor=0.99):
        self.q = q
        self.policy = policy
        self.discount_factor = discount_factor
        self._state = None
        self._action = None

    def act(self, state):
        action = self.policy.no_grad(state)
        self._train(state.reward, state, action)
        self._state = state
        self._action = action
        return action

    def eval(self, state):
        return self.policy.eval(state)

    def _train(self, reward, next_state, next_action):
        if self._state:
            # forward pass
            value = self.q(self._state, self._action)
            # compute target
            target = reward + self.discount_factor * self.q.target(next_state, next_action)
            # compute loss
            loss = mse_loss(value, target)
            # backward pass
            self.q.reinforce(loss)


VSarsaTestAgent = VQNTestAgent
