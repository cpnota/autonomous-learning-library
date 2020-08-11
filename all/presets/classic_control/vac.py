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
        # Model construction
        feature_model_constructor=fc_relu_features,
        value_model_constructor=fc_value_head,
        policy_model_constructor=fc_policy_head
):
    """
    Vanilla Actor-Critic classic control preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        lr_v (float): Learning rate for value network.
        lr_pi (float): Learning rate for policy network and feature network.
        eps (float): Stability parameters for the Adam optimizer.
        feature_model_constructor (function): The function used to construct the neural feature model.
        value_model_constructor (function): The function used to construct the neural value model.
        policy_model_constructor (function): The function used to construct the neural policy model.
    """
    def _vac(env, writer=DummyWriter()):
        value_model = value_model_constructor().to(device)
        policy_model = policy_model_constructor(env).to(device)
        feature_model = feature_model_constructor(env).to(device)

        value_optimizer = Adam(value_model.parameters(), lr=lr_v, eps=eps)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi, eps=eps)
        feature_optimizer = Adam(feature_model.parameters(), lr=lr_pi, eps=eps)

        v = VNetwork(value_model, value_optimizer, writer=writer)
        policy = SoftmaxPolicy(policy_model, policy_optimizer, writer=writer)
        features = FeatureNetwork(feature_model, feature_optimizer)

        return VAC(features, v, policy, discount_factor=discount_factor)
    return _vac
