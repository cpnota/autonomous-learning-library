import copy
from torch.optim import Adam
from all.agents import A2C, A2CTestAgent
from all.approximation import VNetwork, FeatureNetwork
from all.logging import DummyWriter
from all.policies import SoftmaxPolicy
from .models import fc_relu_features, fc_policy_head, fc_value_head
from ..builder import preset_builder
from ..preset import Preset


default_hyperparameters = {
    # Common settings
    "discount_factor": 0.99,
    # Adam optimizer settings
    "lr": 3e-3,
    # Other optimization settings
    "clip_grad": 0.1,
    "entropy_loss_scaling": 0.001,
    "value_loss_scaling": 0.5,
    # Batch settings
    "n_envs": 4,
    "n_steps": 32,
    # Model construction
    "feature_model_constructor": fc_relu_features,
    "value_model_constructor": fc_value_head,
    "policy_model_constructor": fc_policy_head
}


class A2CClassicControlPreset(Preset):
    """
    Advantaged Actor-Critic (A2C) classic control preset.

    Args:
        env (all.environments.GymEnvironment): The classic control environment for which to construct the agent.
        device (torch.device, optional): the device on which to load the agent

    Keyword Args:
        discount_factor (float): Discount factor for future rewards.
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

    def __init__(self, env, device="cuda", **hyperparameters):
        hyperparameters = {**default_hyperparameters, **hyperparameters}
        super().__init__(n_envs=hyperparameters['n_envs'])
        self.value_model = hyperparameters['value_model_constructor']().to(device)
        self.policy_model = hyperparameters['policy_model_constructor'](env).to(device)
        self.feature_model = hyperparameters['feature_model_constructor'](env).to(device)
        self.hyperparameters = hyperparameters
        self.device = device

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        feature_optimizer = Adam(self.feature_model.parameters(), lr=self.hyperparameters["lr"])
        value_optimizer = Adam(self.value_model.parameters(), lr=self.hyperparameters["lr"])
        policy_optimizer = Adam(self.policy_model.parameters(), lr=self.hyperparameters["lr"])

        features = FeatureNetwork(
            self.feature_model,
            feature_optimizer,
            clip_grad=self.hyperparameters["clip_grad"]
        )

        v = VNetwork(
            self.value_model,
            value_optimizer,
            clip_grad=self.hyperparameters["clip_grad"],
            writer=writer
        )

        policy = SoftmaxPolicy(
            self.policy_model,
            policy_optimizer,
            clip_grad=self.hyperparameters["clip_grad"],
            writer=writer
        )

        return A2C(
            features,
            v,
            policy,
            n_envs=self.hyperparameters["n_envs"],
            n_steps=self.hyperparameters["n_steps"],
            discount_factor=self.hyperparameters["discount_factor"],
            entropy_loss_scaling=self.hyperparameters["entropy_loss_scaling"],
            writer=writer
        )

    def test_agent(self):
        features = FeatureNetwork(copy.deepcopy(self.feature_model))
        policy = SoftmaxPolicy(copy.deepcopy(self.policy_model))
        return A2CTestAgent(features, policy)


a2c = preset_builder('a2c', default_hyperparameters, A2CClassicControlPreset)
