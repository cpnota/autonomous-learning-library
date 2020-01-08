# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import PPO
from all.approximation import VNetwork, FeatureNetwork
from all.bodies import TimeFeature
from all.logging import DummyWriter
from all.optim import LinearScheduler
from all.policies import GaussianPolicy
from .models import fc_actor_critic


def ppo(
        # Common settings
        device=torch.device('cuda'),
        discount_factor=0.98,
        lam=0.95,
        last_frame=2e6,
        # Adam optimizer settings
        lr=3e-4,  # Adam learning rate
        eps=1e-5,  # Adam stability
        # Loss scaling
        entropy_loss_scaling=0.01,
        value_loss_scaling=0.5,
        # Training settings
        clip_final=0.01,
        clip_grad=0.5,
        clip_initial=0.2,
        epochs=20,
        minibatches=4,
        # Batch settings
        n_envs=32,
        n_steps=128,
):
    def _ppo(envs, writer=DummyWriter()):
        final_anneal_step = last_frame * epochs * minibatches / (n_steps * n_envs)
        env = envs[0]

        feature_model, value_model, policy_model = fc_actor_critic(env)
        feature_model.to(device)
        value_model.to(device)
        policy_model.to(device)

        feature_optimizer = Adam(
            feature_model.parameters(), lr=lr, eps=eps
        )
        value_optimizer = Adam(value_model.parameters(), lr=lr, eps=eps)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr, eps=eps)

        features = FeatureNetwork(
            feature_model,
            feature_optimizer,
            clip_grad=clip_grad,
            scheduler=CosineAnnealingLR(
                feature_optimizer,
                final_anneal_step
            ),
            writer=writer
        )
        v = VNetwork(
            value_model,
            value_optimizer,
            loss_scaling=value_loss_scaling,
            clip_grad=clip_grad,
            writer=writer,
            scheduler=CosineAnnealingLR(
                value_optimizer,
                final_anneal_step
            ),
        )
        policy = GaussianPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
            clip_grad=clip_grad,
            writer=writer,
            scheduler=CosineAnnealingLR(
                policy_optimizer,
                final_anneal_step
            ),
        )

        return TimeFeature(PPO(
            features,
            v,
            policy,
            epsilon=LinearScheduler(
                clip_initial,
                clip_final,
                0,
                final_anneal_step,
                name='clip',
                writer=writer
            ),
            epochs=epochs,
            minibatches=minibatches,
            n_envs=n_envs,
            n_steps=n_steps,
            discount_factor=discount_factor,
            lam=lam,
            entropy_loss_scaling=entropy_loss_scaling,
            writer=writer,
        ))

    return _ppo, n_envs


__all__ = ["ppo"]
