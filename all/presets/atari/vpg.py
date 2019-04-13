import torch
from torch import nn, optim
from all.agents import VPG
from all.approximation import ValueNetwork
from all.bodies import DeepmindAtariBody
from all.experiments import DummyWriter
from all.layers import Flatten
from all.policies import SoftmaxPolicy


def conv_features():
    return nn.Sequential(
        nn.Conv2d(4, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1),
        nn.ReLU(),
        Flatten()
    )


def value_net(features):
    return nn.Sequential(
        features,
        nn.Linear(3456, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
    )


def policy_net(env, features):
    return nn.Sequential(
        features,
        nn.Linear(3456, 512),
        nn.ReLU(),
        nn.Linear(512, env.action_space.n)
    )


def vpg(
        gamma=0.99,
        lr_v=1e-2,
        lr_pi=1e-2,
        n_episodes=5,
        device=torch.device('cpu')
):
    def _vpg_atari(env, writer=DummyWriter()):
        features = conv_features()
        value_model = value_net(features).to(device)
        policy_model = policy_net(env, features).to(device)
        value_optimizer = optim.Adam(value_model.parameters(), lr=lr_v)
        policy_optimizer = optim.Adam(policy_model.parameters(), lr=lr_pi)
        v = ValueNetwork(value_model, value_optimizer, writer=writer)
        policy = SoftmaxPolicy(
            policy_model,
            policy_optimizer,
            env.action_space.n,
            writer=writer
        )

        def weights_init(layer):
            if isinstance(layer, nn.Linear):
                if layer.out_features == env.action_space.n:
                    nn.init.zeros_(layer.weight.data)
                    nn.init.zeros_(layer.bias.data)

        policy_model.apply(weights_init)

        return DeepmindAtariBody(
            VPG(v, policy, gamma=gamma, n_episodes=n_episodes),
            env
        )
    return _vpg_atari


__all__ = ["vpg"]
