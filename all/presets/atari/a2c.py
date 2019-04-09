# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch import nn
from torch.optim import Adam
from all.layers import Flatten, Linear0
from all.agents import A2C
from all.bodies import ParallelAtariBody
from all.approximation import ValueNetwork, FeatureNetwork
from all.policies import SoftmaxPolicy


def conv_features():
    return nn.Sequential(
        nn.Conv2d(4, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        Flatten()
    )


def value_net():
    return nn.Sequential(
        nn.Linear(3456, 512),
        nn.ReLU(),
        Linear0(512, 1)
    )


def policy_net(env):
    return nn.Sequential(
        nn.Linear(3456, 512),
        nn.ReLU(),
        Linear0(512, env.action_space.n)
    )


def a2c(
        batch_size=64,
        clip_grad=0.1,
        discount_factor=0.99,
        entropy_loss_scaling=0.01,
        eps=1.5e-4,  # Adam epsilon
        lr=1e-3,
        n_steps=4,
        device=torch.device('cpu')
):
    def _a2c(envs):
        env = envs[0]
        feature_model = conv_features().to(device)
        value_model = value_net().to(device)
        policy_model = policy_net(env).to(device)

        feature_optimizer = Adam(feature_model.parameters(), lr=lr, eps=eps)
        value_optimizer = Adam(value_model.parameters(), lr=lr, eps=eps)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr, eps=eps)

        features = FeatureNetwork(
            feature_model, feature_optimizer, clip_grad=clip_grad)
        v = ValueNetwork(value_model, value_optimizer, clip_grad=clip_grad)
        policy = SoftmaxPolicy(
            policy_model,
            policy_optimizer,
            env.action_space.n,
            entropy_loss_scaling=entropy_loss_scaling,
            clip_grad=clip_grad
        )
        return ParallelAtariBody(
            A2C(
                features,
                v,
                policy,
                n_steps=n_steps,
                batch_size=batch_size,
                discount_factor=discount_factor
            ),
            envs
        )
    return _a2c


__all__ = ["a2c"]
