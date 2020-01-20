from torch.optim import Adam
from all.agents import VAC
from all.approximation import VNetwork, FeatureNetwork
from all.logging import DummyWriter
from all.policies import SoftmaxPolicy
from .models import fc_relu_features, fc_policy_head, fc_value_head


def vac(
        # Common settings
        device="cpu",
        discount_factor=0.99,
        # Adam optimizer settings
        lr_v=5e-3,
        lr_pi=1e-3,
        eps=1e-5,
):
    """
    Vanilla Actor-Critic classic control preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        lr_v (float): Learning rate for value network.
        lr_pi (float): Learning rate for policy network and feature network.
        eps (float): Stability parameters for the Adam optimizer.
    """
    def _vac(env, writer=DummyWriter()):
        value_model = fc_value_head().to(device)
        policy_model = fc_policy_head(env).to(device)
        feature_model = fc_relu_features(env).to(device)

        value_optimizer = Adam(value_model.parameters(), lr=lr_v, eps=eps)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi, eps=eps)
        feature_optimizer = Adam(feature_model.parameters(), lr=lr_pi, eps=eps)

        v = VNetwork(value_model, value_optimizer, writer=writer)
        policy = SoftmaxPolicy(policy_model, policy_optimizer, writer=writer)
        features = FeatureNetwork(feature_model, feature_optimizer)

        return VAC(features, v, policy, discount_factor=discount_factor)
    return _vac
