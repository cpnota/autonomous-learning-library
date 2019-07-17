# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import RMSprop
from all import nn
from all.agents import PPO
from all.bodies import ParallelAtariBody
from all.approximation import VNetwork, FeatureNetwork
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
        nn.Flatten(),
    )


def value_net():
    return nn.Sequential(nn.Linear(3456, 512), nn.ReLU(), nn.Linear0(512, 1))


def policy_net(env):
    return nn.Sequential(
        nn.Linear(3456, 512), nn.ReLU(), nn.Linear0(512, env.action_space.n)
    )


def ppo(
        # stable baselines hyperparameters
        clip_grad=0.5,
        discount_factor=0.99,
        lam=0.95,   # GAE lambda (similar to e-traces)
        lr=2.5e-4,  # RMSprop learning rate
        alpha=0.99, # RMSprop momentum decay
        eps=1e-4,   # RMSprop stability
        entropy_loss_scaling=0.02,
        value_loss_scaling=0.25,
        feature_lr_scaling=1,
        epochs=4,
        minibatches=4,
        epsilon=0.2,
        n_envs=16,
        n_steps=128,
        device=torch.device("cpu"),
):
    def _ppo(envs, writer=DummyWriter()):
        env = envs[0]

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

        return ParallelAtariBody(
            PPO(
                features,
                v,
                policy,
                epsilon=epsilon,
                epochs=epochs,
                minibatches=minibatches,
                n_envs=n_envs,
                n_steps=n_steps,
                discount_factor=discount_factor,
                lam=lam
            ),
            envs,
        )

    return _ppo, n_envs


__all__ = ["ppo"]
