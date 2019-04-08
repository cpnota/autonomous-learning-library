# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch import nn
from torch.optim import Adam
from all.layers import Flatten
from all.agents import A2C
from all.approximation import ValueNetwork
from all.policies import SoftmaxPolicy


def fc_value(env):
    return nn.Sequential(
        Flatten(),
        nn.Linear(env.state_space.shape[0], 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )

def fc_policy(env):
    return nn.Sequential(
        Flatten(),
        nn.Linear(env.state_space.shape[0], 256),
        nn.ReLU(),
        nn.Linear(256, env.action_space.n)
    )

def a2c(
        lr_v=5e-3,
        lr_pi=1e-3,
        n_steps=16,
        discount_factor=0.99,
):
    def _a2c(env):
        value_model = fc_value(env)
        value_optimizer = Adam(value_model.parameters(), lr=lr_v)
        v = ValueNetwork(value_model, value_optimizer)
        policy_model = fc_policy(env)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        policy = SoftmaxPolicy(policy_model, policy_optimizer, env.action_space.n)
        return A2C(v, policy, n_steps=n_steps, discount_factor=discount_factor)
    return _a2c

__all__ = ["a2c"]
 