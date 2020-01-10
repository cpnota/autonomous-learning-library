from torch.optim import Adam
from all.agents import A2C
from all.approximation import VNetwork, FeatureNetwork
from all.logging import DummyWriter
from all.policies import SoftmaxPolicy
from .models import fc_relu_features, fc_policy_head, fc_value_head


def a2c(
        # Common settings
        device="cpu",
        discount_factor=0.99,
        # Adam optimizer settings
        lr=3e-3,
        # Other optimization settings
        clip_grad=0.1,
        entropy_loss_scaling=0.001,
        # Batch settings
        n_envs=4,
        n_steps=32,
):
    """
    A2C classic control preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        clip_grad (float): The maximum magnitude of the gradient for any given parameter. Set to 0 to disable.
        entropy_loss_scaling (float): Coefficient for the entropy term in the total loss.
        n_envs (int): Number of parallel environments.
        n_steps (int): Length of each rollout.
    """
    def _a2c(envs, writer=DummyWriter()):
        env = envs[0]
        feature_model = fc_relu_features(env).to(device)
        value_model = fc_value_head().to(device)
        policy_model = fc_policy_head(env).to(device)

        feature_optimizer = Adam(feature_model.parameters(), lr=lr)
        value_optimizer = Adam(value_model.parameters(), lr=lr)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr)

        features = FeatureNetwork(
            feature_model, feature_optimizer, clip_grad=clip_grad)
        v = VNetwork(
            value_model,
            value_optimizer,
            clip_grad=clip_grad,
            writer=writer
        )
        policy = SoftmaxPolicy(
            policy_model,
            policy_optimizer,
            clip_grad=clip_grad,
            writer=writer
        )
        return A2C(
            features,
            v,
            policy,
            n_envs=n_envs,
            n_steps=n_steps,
            discount_factor=discount_factor,
            entropy_loss_scaling=entropy_loss_scaling,
            writer=writer
        )
    return _a2c, n_envs


__all__ = ["a2c"]
