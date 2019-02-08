from torch import nn
from torch.optim import Adam
from all.layers import Flatten
from all.agents import REINFORCE
from all.approximation import ValueNetwork
from all.policies import SoftmaxPolicy


def conv_net(outputs):
    return nn.Sequential(
        nn.Conv2d(4, 16, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(2592, 256),
        nn.ReLU(),
        nn.Linear(256, outputs)
    )


def reinforce_atari(
        lr_v=1e-5,
        lr_pi=1e-5
        ):
    def _reinforce_atari(env):
        value_model = conv_net(1)
        value_optimizer = Adam(value_model.parameters(), lr=lr_v)
        v = ValueNetwork(value_model, value_optimizer)
        policy_model = conv_net(env.action_space.n)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        policy = SoftmaxPolicy(policy_model, policy_optimizer)
        return REINFORCE(v, policy)
    return _reinforce_atari


__all__ = ["reinforce_atari"]
