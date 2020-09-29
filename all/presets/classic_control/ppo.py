from torch.optim import Adam
from all.agents import PPO
from all.approximation import VNetwork, FeatureNetwork
from all.logging import DummyWriter
from all.policies import SoftmaxPolicy
from .models import fc_relu_features, fc_policy_head, fc_value_head

def ppo(
        # Common settings
        device="cpu",
        discount_factor=0.99,
        # Adam optimizer settings
        lr=1e-3,
        # Other optimization settings
        clip_grad=0.1,
        entropy_loss_scaling=0.001,
        epsilon=0.2,
        # Batch settings
        epochs=4,
        minibatches=4,
        n_envs=8,
        n_steps=8,
        # GAE settings
        lam=0.95,
        # Model construction
        feature_model_constructor=fc_relu_features,
        value_model_constructor=fc_value_head,
        policy_model_constructor=fc_policy_head
):
    """
    PPO classic control preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        clip_grad (float): The maximum magnitude of the gradient for any given parameter.
            Set to 0 to disable.
        entropy_loss_scaling (float): Coefficient for the entropy term in the total loss.
        epsilon (float): Value for epsilon in the clipped PPO objective function.
        epochs (int): Number of times to iterature through each batch.
        minibatches (int): The number of minibatches to split each batch into.
        n_envs (int): Number of parallel actors.
        n_steps (int): Length of each rollout.
        lam (float): The Generalized Advantage Estimate (GAE) decay parameter.
        feature_model_constructor (function): The function used to construct the neural feature model.
        value_model_constructor (function): The function used to construct the neural value model.
        policy_model_constructor (function): The function used to construct the neural policy model.
    """
    def _ppo(envs, writer=DummyWriter()):
        env = envs[0]
        feature_model = feature_model_constructor(env).to(device)
        value_model = value_model_constructor().to(device)
        policy_model = policy_model_constructor(env).to(device)

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
        return PPO(
            features,
            v,
            policy,
            epsilon=epsilon,
            epochs=epochs,
            lam=lam,
            minibatches=minibatches,
            n_envs=n_envs,
            n_steps=n_steps,
            discount_factor=discount_factor,
            entropy_loss_scaling=entropy_loss_scaling,
            writer=writer
        )
    return _ppo, n_envs

__all__ = ["ppo"]
