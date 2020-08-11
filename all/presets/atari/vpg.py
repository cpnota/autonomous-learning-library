from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import VPG
from all.approximation import VNetwork, FeatureNetwork
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.policies import SoftmaxPolicy
from .models import nature_features, nature_value_head, nature_policy_head


def vpg(
        # Common settings
        device="cuda",
        discount_factor=0.99,
        last_frame=40e6,
        # Adam optimizer settings
        lr=7e-4,
        eps=1.5e-4,
        # Other optimization settings
        clip_grad=0.5,
        value_loss_scaling=0.25,
        min_batch_size=1000,
        # Model construction
        feature_model_constructor=nature_features,
        value_model_constructor=nature_value_head,
        policy_model_constructor=nature_policy_head
):
    """
    Vanilla Policy Gradient Atari preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        last_frame (int): Number of frames to train.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        clip_grad (float): The maximum magnitude of the gradient for any given parameter.
            Set to 0 to disable.
        value_loss_scaling (float): Coefficient for the value function loss.
        min_batch_size (int): Continue running complete episodes until at least this many
            states have been seen since the last update.
        feature_model_constructor (function): The function used to construct the neural feature model.
        value_model_constructor (function): The function used to construct the neural value model.
        policy_model_constructor (function): The function used to construct the neural policy model.
    """
    final_anneal_step = last_frame / (min_batch_size * 4)

    def _vpg_atari(env, writer=DummyWriter()):
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
            VPG(features, v, policy, discount_factor=discount_factor, min_batch_size=min_batch_size),
            episodic_lives=True
        )
    return _vpg_atari


__all__ = ["vpg"]
