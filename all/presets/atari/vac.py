import torch
from torch.optim import RMSprop
from all.agents import VAC
from all.approximation import VNetwork, FeatureNetwork
from all.bodies import ParallelAtariBody
from all.experiments import DummyWriter
from all.policies import SoftmaxPolicy
from .models import nature_cnn, nature_value_head, nature_policy_head


def vac(
        # taken from stable-baselines a2c
        discount_factor=0.99,
        value_loss_scaling=0.25,
        entropy_loss_scaling=0.01,
        clip_grad=0.5,
        lr=7e-4,    # RMSprop learning rate
        alpha=0.99, # RMSprop momentum decay
        eps=1e-5,   # RMSprop stability
        n_envs=16,
        device=torch.device('cpu')
):
    '''Vanilla Actor-Critic Atari preset'''
    def _vac(envs, writer=DummyWriter()):
        value_model = nature_value_head().to(device)
        policy_model = nature_policy_head(envs[0]).to(device)
        feature_model = nature_cnn().to(device)

        value_optimizer = RMSprop(value_model.parameters(), alpha=alpha, lr=lr, eps=eps)
        policy_optimizer = RMSprop(
            policy_model.parameters(), alpha=alpha, lr=lr, eps=eps
        )
        feature_optimizer = RMSprop(
            feature_model.parameters(), alpha=alpha, lr=lr, eps=eps
        )

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
            envs[0].action_space.n,
            entropy_loss_scaling=entropy_loss_scaling,
            clip_grad=clip_grad,
            writer=writer,
        )
        features = FeatureNetwork(feature_model, feature_optimizer, clip_grad=clip_grad)

        return ParallelAtariBody(
            VAC(features, v, policy, gamma=discount_factor),
            envs
        )
    return _vac, n_envs
