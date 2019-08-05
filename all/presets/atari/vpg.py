import torch
from torch.optim import RMSprop
from all.agents import VPG
from all.approximation import VNetwork, FeatureNetwork
from all.bodies import RewardClipping
from all.logging import DummyWriter
from all.policies import SoftmaxPolicy
from .models import nature_features, nature_value_head, nature_policy_head


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
        min_batch_size=1000,
        device=torch.device('cpu')
):
    def _vpg_atari(env, writer=DummyWriter()):
        feature_model = nature_features().to(device)
        value_model = nature_value_head().to(device)
        policy_model = nature_policy_head(env).to(device)

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
            clip_grad=clip_grad,
            writer=writer
        )
        v = VNetwork(
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

        return RewardClipping(
            VPG(features, v, policy, gamma=discount_factor, min_batch_size=min_batch_size),
        )
    return _vpg_atari


__all__ = ["vpg"]
