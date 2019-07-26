# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from all.agents import VPG
from all.approximation import VNetwork, FeatureNetwork
from all.experiments import DummyWriter
from all.policies import SoftmaxPolicy
from .models import fc_relu_features, fc_policy_head, fc_value_head


def vpg(
        clip_grad=0,
        entropy_loss_scaling=0.001,
        gamma=0.99,
        lr=5e-3,
        min_batch_size=500,
        device=torch.device('cpu')
):
    def _vpg(env, writer=DummyWriter()):
        feature_model = fc_relu_features(env).to(device)
        value_model = fc_value_head().to(device)
        policy_model = fc_policy_head(env).to(device)

        feature_optimizer = Adam(feature_model.parameters(), lr=lr)
        value_optimizer = Adam(value_model.parameters(), lr=lr)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr)

        features = FeatureNetwork(
            feature_model,
            feature_optimizer,
            clip_grad=clip_grad,
            writer=writer
        )
        v = VNetwork(
            value_model,
            value_optimizer,
            clip_grad=clip_grad,
            writer=writer
        )
        policy = SoftmaxPolicy(
            policy_model,
            policy_optimizer,
            env.action_space.n,
            entropy_loss_scaling=entropy_loss_scaling,
            clip_grad=clip_grad,
            writer=writer
        )
        return VPG(features, v, policy, gamma=gamma, min_batch_size=min_batch_size)
    return _vpg


__all__ = ["vpg"]
