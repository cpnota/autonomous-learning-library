import torch
from torch.nn.functional import mse_loss
from ._agent import Agent
from ._parallel_agent import ParallelAgent
from .dqn import DQNTestAgent


class VQN(ParallelAgent):
    '''
    Vanilla Q-Network (VQN).
    VQN is an implementation of the Q-learning algorithm found in the Sutton and Barto (2018) textbook.
    Q-learning algorithms attempt to learning the optimal policy while executing a (generally)
    suboptimal policy (typically epsilon-greedy). In theory, This allows the agent to gain the benefits
    of exploration without sacrificing the performance of the final policy. However, the cost of this
    is that Q-learning is generally less stable than its on-policy bretheren, SARSA.
    http://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf

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
        self._train(state.reward, state)
        action = self.policy.no_grad(state)
        self._state = state
        self._action = action
        return action

    def eval(self, state):
        return self.policy.eval(state)

    def _train(self, reward, next_state):
        if self._state:
            # forward pass
            value = self.q(self._state, self._action)
            # compute target
            target = reward + self.discount_factor * torch.max(self.q.target(next_state), dim=1)[0]
            # compute loss
            loss = mse_loss(value, target)
            # backward pass
            self.q.reinforce(loss)


class VQNTestAgent(Agent, ParallelAgent):
    def __init__(self, policy):
        self.policy = policy

    def act(self, state):
        return self.policy.eval(state)
