# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch import nn
from torch.optim import Adam
from all.layers import Flatten
from all.agents import ActorCritic
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


def ac_cc(
        lr_v=1e-4,
        lr_pi=1e-4
):
    def _ac_cc(env):
        value_model = fc_value(env)
        value_optimizer = Adam(value_model.parameters(), lr=lr_v)
        v = ValueNetwork(value_model, value_optimizer)
        policy_model = fc_policy(env)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        policy = SoftmaxPolicy(policy_model, policy_optimizer)
        return ActorCritic(v, policy)
    return _ac_cc


__all__ = ["ac_cc"]
