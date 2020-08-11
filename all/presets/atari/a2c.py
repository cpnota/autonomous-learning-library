from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import A2C
from all.bodies import DeepmindAtariBody
from all.approximation import VNetwork, FeatureNetwork
from all.logging import DummyWriter
from all.policies import SoftmaxPolicy
from .models import nature_features, nature_value_head, nature_policy_head


def a2c(
        # Common settings
        device="cuda",
        discount_factor=0.99,
        last_frame=40e6,
        # Adam optimizer settings
        lr=7e-4,
        eps=1.5e-4,
        # Other optimization settings
        clip_grad=0.1,
        entropy_loss_scaling=0.01,
        value_loss_scaling=0.5,
        # Batch settings
        n_envs=16,
        n_steps=5,
        # Model construction
        feature_model_constructor=nature_features,
        value_model_constructor=nature_value_head,
        policy_model_constructor=nature_policy_head
):
    """
    A2C Atari preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        last_frame (int): Number of frames to train.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        clip_grad (float): The maximum magnitude of the gradient for any given parameter.
            Set to 0 to disable.
        entropy_loss_scaling (float): Coefficient for the entropy term in the total loss.
        value_loss_scaling (float): Coefficient for the value function loss.
        n_envs (int): Number of parallel environments.
        n_steps (int): Length of each rollout.
        feature_model_constructor (function): The function used to construct the neural feature model.
        value_model_constructor (function): The function used to construct the neural value model.
        policy_model_constructor (function): The function used to construct the neural policy model.
    """
    def _a2c(envs, writer=DummyWriter()):
        env = envs[0]
        final_anneal_step = last_frame / (n_steps * n_envs * 4)

        value_model = value_model_constructor().to(device)
        policy_model = policy_model_constructor(env).to(device)
        feature_model = feature_model_constructor().to(device)

        feature_optimizer = Adam(feature_model.parameters(), lr=lr, eps=eps)
        value_optimizer = Adam(value_model.parameters(), lr=lr, eps=eps)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr, eps=eps)

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
            scheduler=CosineAnnealingLR(
                policy_optimizer,
                final_anneal_step,
            ),
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
                entropy_loss_scaling=entropy_loss_scaling,
                writer=writer
            ),
        )

    return _a2c, n_envs


__all__ = ["a2c"]
