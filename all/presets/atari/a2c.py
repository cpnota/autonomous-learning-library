# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import A2C
from all.bodies import DeepmindAtariBody
from all.approximation import VNetwork, FeatureNetwork
from all.logging import DummyWriter
from all.policies import SoftmaxPolicy
from .models import nature_features, nature_value_head, nature_policy_head


def a2c(
        # taken from stable-baselines
        alpha=0.99,  # RMSprop momentum decay
        clip_grad=0.5,
        discount_factor=0.99,
        entropy_loss_scaling=0.01,
        eps=1e-5,  # RMSprop stability
        final_frame=40e6, # Anneal LR and clip until here
        lr=7e-4,  # RMSprop learning rate
        n_envs=16,
        n_steps=5,
        value_loss_scaling=0.25,
        device=torch.device("cuda"),
):
    final_anneal_step = final_frame / (n_steps * n_envs * 4)
    def _a2c(envs, writer=DummyWriter()):
        env = envs[0]

        value_model = nature_value_head().to(device)
        policy_model = nature_policy_head(envs[0]).to(device)
        feature_model = nature_features().to(device)

        feature_optimizer = RMSprop(
            feature_model.parameters(), alpha=alpha, lr=lr, eps=eps
        )
        value_optimizer = RMSprop(value_model.parameters(), alpha=alpha, lr=lr, eps=eps)
        policy_optimizer = RMSprop(
            policy_model.parameters(), alpha=alpha, lr=lr, eps=eps
        )

        features = FeatureNetwork(
            feature_model,
            feature_optimizer,
            scheduler=CosineAnnealingLR(
                feature_optimizer,
                final_anneal_step,
            ),
            clip_grad=clip_grad,
            writer=writer
        )
        v = VNetwork(
            value_model,
            value_optimizer,
            scheduler=CosineAnnealingLR(
                value_optimizer,
                final_anneal_step,
            ),
            loss_scaling=value_loss_scaling,
            clip_grad=clip_grad,
            writer=writer
        )
        policy = SoftmaxPolicy(
            policy_model,
            policy_optimizer,
            env.action_space.n,
            scheduler=CosineAnnealingLR(
                policy_optimizer,
                final_anneal_step,
            ),
            entropy_loss_scaling=entropy_loss_scaling,
            clip_grad=clip_grad,
            writer=writer
        )

        return DeepmindAtariBody(
            A2C(
                features,
                v,
                policy,
                n_envs=n_envs,
                n_steps=n_steps,
                discount_factor=discount_factor,
            ),
        )

    return _a2c, n_envs


__all__ = ["a2c"]
