# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import PPO
from all import nn
from all.approximation import VNetwork, FeatureNetwork
from all.logging import DummyWriter
from all.optim import LinearScheduler
from all.policies import GaussianPolicy

def fc_features(env):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0], 64),
        nn.ReLU(),
    )

def fc_v():
    return nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

def fc_policy(env):
    return nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, env.action_space.shape[0] * 2)
    )

def ppo(
        clip_grad=0.5,
        discount_factor=0.99,
        lam=0.95,  # GAE lambda (similar to e-traces)
        lr=3e-4,  # Adam learning rate
        eps=1e-5,  # Adam stability
        entropy_loss_scaling=0.01,
        value_loss_scaling=0.5,
        clip_initial=0.1,
        clip_final=0.01,
        final_frame=2e6, # Anneal LR and clip until here
        epochs=10,
        minibatches=4,
        n_envs=32,
        n_steps=128,
        device=torch.device("cuda"),
):
    # Update epoch * minibatches times per update,
    # but we only update once per n_steps
    final_anneal_step = final_frame * epochs * minibatches / (n_steps * n_envs)

    def _ppo(envs, writer=DummyWriter()):
        env = envs[0]

        value_model = fc_v().to(device)
        policy_model = fc_policy(env).to(device)
        feature_model = fc_features(env).to(device)

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

        return PPO(
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
        )

    return _ppo, n_envs


__all__ = ["ppo"]
