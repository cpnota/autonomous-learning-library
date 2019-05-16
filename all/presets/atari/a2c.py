# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch import nn
from torch.optim import RMSprop
from all.layers import Flatten, Linear0
from all.agents import A2C
from all.bodies import ParallelAtariBody
from all.approximation import ValueNetwork, FeatureNetwork
from all.experiments import DummyWriter
from all.policies import SoftmaxPolicy


def conv_features():
    return nn.Sequential(
        nn.Conv2d(4, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        Flatten(),
    )


def value_net():
    return nn.Sequential(nn.Linear(3456, 512), nn.ReLU(), Linear0(512, 1))


def policy_net(env):
    return nn.Sequential(
        nn.Linear(3456, 512), nn.ReLU(), Linear0(512, env.action_space.n)
    )


def a2c(
    clip_grad=0.1,
    discount_factor=0.99,
    entropy_loss_scaling=0.01,
    alpha=0.99,  # RMSprop alpha
    eps=1e-4,  # RMSprop epsilon
    lr=1e-3,
    feature_lr_scaling=0.25,
    n_envs=16,
    n_steps=5,
    device=torch.device("cpu"),
):
    def _a2c(envs, writer=DummyWriter()):
        env = envs[0]
        feature_model = conv_features().to(device)
        value_model = value_net().to(device)
        policy_model = policy_net(env).to(device)

        feature_optimizer = RMSprop(
            feature_model.parameters(), alpha=alpha, lr=lr * feature_lr_scaling, eps=eps
        )
        value_optimizer = RMSprop(value_model.parameters(), alpha=alpha, lr=lr, eps=eps)
        policy_optimizer = RMSprop(
            policy_model.parameters(), alpha=alpha, lr=lr, eps=eps
        )

        features = FeatureNetwork(feature_model, feature_optimizer, clip_grad=clip_grad)
        v = ValueNetwork(
            value_model, value_optimizer, clip_grad=clip_grad, writer=writer
        )
        policy = SoftmaxPolicy(
            policy_model,
            policy_optimizer,
            env.action_space.n,
            entropy_loss_scaling=entropy_loss_scaling,
            clip_grad=clip_grad,
            writer=writer,
        )
        return ParallelAtariBody(
            A2C(
                features,
                v,
                policy,
                n_envs=n_envs,
                n_steps=n_steps,
                discount_factor=discount_factor,
            ),
            envs,
        )

    return _a2c, n_envs


__all__ = ["a2c"]
