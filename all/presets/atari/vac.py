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
        # Model construction
        feature_model_constructor=nature_features,
        value_model_constructor=nature_value_head,
        policy_model_constructor=nature_policy_head
):
    """
    Vanilla Actor-Critic Atari preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        last_frame (int): Number of frames to train.
        lr_v (float): Learning rate for value network.
        lr_pi (float): Learning rate for policy network and feature network.
        eps (float): Stability parameters for the Adam optimizer.
        clip_grad (float): The maximum magnitude of the gradient for any given parameter.
            Set to 0 to disable.
        value_loss_scaling (float): Coefficient for the value function loss.
        n_envs (int): Number of parallel environments.
        feature_model_constructor (function): The function used to construct the neural feature model.
        value_model_constructor (function): The function used to construct the neural value model.
        policy_model_constructor (function): The function used to construct the neural policy model.
    """
    def _vac(envs, writer=DummyWriter()):
        value_model = value_model_constructor().to(device)
        policy_model = policy_model_constructor(envs[0]).to(device)
        feature_model = feature_model_constructor().to(device)

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
            VAC(features, v, policy, discount_factor=discount_factor),
        )
    return _vac, n_envs
