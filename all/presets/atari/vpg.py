import torch
from torch import nn
from torch.optim import RMSprop
from all.agents import VPG
from all.approximation import ValueNetwork, FeatureNetwork
from all.bodies import DeepmindAtariBody
from all.experiments import DummyWriter
from all.layers import Flatten, Linear0
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


def vpg(
        # match a2c hypers
        clip_grad=0.5,
        discount_factor=0.99,
        lr=7e-4,    # RMSprop learning rate
        alpha=0.99, # RMSprop momentum decay
        eps=1e-4,   # RMSprop stability
        entropy_loss_scaling=0.01,
        value_loss_scaling=0.25,
        feature_lr_scaling=1,
        n_episodes=5,
        device=torch.device('cpu')
):
    def _vpg_atari(env, writer=DummyWriter()):
        feature_model = conv_features().to(device)
        value_model = value_net().to(device)
        policy_model = policy_net(env).to(device)

        feature_optimizer = RMSprop(
            feature_model.parameters(),
            alpha=alpha,
            lr=lr * feature_lr_scaling,
            eps=eps
        )
        value_optimizer = RMSprop(
            value_model.parameters(),
            alpha=alpha,
            lr=lr,
            eps=eps
        )
        policy_optimizer = RMSprop(
            policy_model.parameters(),
            alpha=alpha,
            lr=lr,
            eps=eps
        )

        features = FeatureNetwork(
            feature_model,
            feature_optimizer,
            clip_grad=clip_grad
        )
        v = ValueNetwork(
            value_model,
            value_optimizer,
            loss_scaling=value_loss_scaling,
            clip_grad=clip_grad,
            writer=writer
        )
        policy = SoftmaxPolicy(
            policy_model,
            policy_optimizer,
            env.action_space.n,
            entropy_loss_scaling=entropy_loss_scaling,
            clip_grad=clip_grad,
            writer=writer,
        )

        return DeepmindAtariBody(
            VPG(features, v, policy, gamma=discount_factor, n_episodes=n_episodes),
            env
        )
    return _vpg_atari


__all__ = ["vpg"]
