# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import RMSprop
from all.agents import A2C
from all.bodies import ParallelAtariBody
from all.approximation import VNetwork, FeatureNetwork
from all.experiments import DummyWriter
from all.policies import SoftmaxPolicy
from .models import nature_cnn, nature_value_head, nature_policy_head


def a2c(
        # taken from stable-baselines
        discount_factor=0.99,
        n_steps=5,
        value_loss_scaling=0.25,
        entropy_loss_scaling=0.01,
        clip_grad=0.5,
        lr=7e-4,  # RMSprop learning rate
        alpha=0.99,  # RMSprop momentum decay
        eps=1e-5,  # RMSprop stability
        n_envs=16,
        device=torch.device("cpu"),
):
    def _a2c(envs, writer=DummyWriter()):
        env = envs[0]

        value_model = nature_value_head().to(device)
        policy_model = nature_policy_head(envs[0]).to(device)
        feature_model = nature_cnn().to(device)

        feature_optimizer = RMSprop(
            feature_model.parameters(), alpha=alpha, lr=lr, eps=eps
        )
        value_optimizer = RMSprop(value_model.parameters(), alpha=alpha, lr=lr, eps=eps)
        policy_optimizer = RMSprop(
            policy_model.parameters(), alpha=alpha, lr=lr, eps=eps
        )

        features = FeatureNetwork(feature_model, feature_optimizer, clip_grad=clip_grad)
        v = VNetwork(
            value_model,
            value_optimizer,
            loss_scaling=value_loss_scaling,
            clip_grad=clip_grad,
            writer=writer,
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
