from torch.optim import Adam
from all.agents import VAC
from all.approximation import VNetwork, FeatureNetwork
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.policies import SoftmaxPolicy
from .models import nature_features, nature_value_head, nature_policy_head


def vac(
        # Common settings
        device="cuda",
        discount_factor=0.99,
        # Adam optimizer settings
        lr_v=5e-4,
        lr_pi=1e-4,
        eps=1.5e-4,
        # Other optimization settings
        clip_grad=0.5,
        value_loss_scaling=0.25,
        # Parallel actors
        n_envs=16,
):
    '''Vanilla Actor-Critic Atari preset'''
    def _vac(envs, writer=DummyWriter()):
        value_model = nature_value_head().to(device)
        policy_model = nature_policy_head(envs[0]).to(device)
        feature_model = nature_features().to(device)

        value_optimizer = Adam(value_model.parameters(), lr=lr_v, eps=eps)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi, eps=eps)
        feature_optimizer = Adam(feature_model.parameters(), lr=lr_pi, eps=eps)

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
            clip_grad=clip_grad,
            writer=writer,
        )
        features = FeatureNetwork(
            feature_model,
            feature_optimizer,
            clip_grad=clip_grad,
            writer=writer
        )

        return DeepmindAtariBody(
            VAC(features, v, policy, gamma=discount_factor),
        )
    return _vac, n_envs
