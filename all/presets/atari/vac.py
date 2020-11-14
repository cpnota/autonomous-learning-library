import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import VAC, VACTestAgent
from all.approximation import VNetwork, FeatureNetwork
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.policies import SoftmaxPolicy
from .models import nature_features, nature_value_head, nature_policy_head
from ..builder import preset_builder
from ..preset import Preset

default_hyperparameters = {
    # Common settings
    "discount_factor": 0.99,
    # Adam optimizer settings
    "lr_v": 5e-4,
    "lr_pi": 1e-4,
    "eps": 1.5e-4,
    # Other optimization settings
    "clip_grad": 0.5,
    "value_loss_scaling": 0.25,
    # Parallel actors
    "n_envs": 16,
}

class VACAtariPreset(Preset):
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
    def __init__(self, hyperparameters, env, device):
        super().__init__(n_envs=hyperparameters['n_envs'])
        self.value_model = nature_value_head().to(device)
        self.policy_model = nature_policy_head(env).to(device)
        self.feature_model = nature_features().to(device)
        self.hyperparameters = hyperparameters
        self.device = device

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        n_updates = train_steps / self.hyperparameters["n_envs"]

        feature_optimizer = Adam(self.feature_model.parameters(), lr=self.hyperparameters["lr_pi"], eps=self.hyperparameters["eps"])
        value_optimizer = Adam(self.value_model.parameters(), lr=self.hyperparameters["lr_v"], eps=self.hyperparameters["eps"])
        policy_optimizer = Adam(self.policy_model.parameters(), lr=self.hyperparameters["lr_pi"], eps=self.hyperparameters["eps"])

        features = FeatureNetwork(
            self.feature_model,
            feature_optimizer,
            scheduler=CosineAnnealingLR(feature_optimizer, n_updates),
            clip_grad=self.hyperparameters["clip_grad"],
            writer=writer
        )

        v = VNetwork(
            self.value_model,
            value_optimizer,
            scheduler=CosineAnnealingLR(value_optimizer, n_updates),
            loss_scaling=self.hyperparameters["value_loss_scaling"],
            clip_grad=self.hyperparameters["clip_grad"],
            writer=writer
        )

        policy = SoftmaxPolicy(
            self.policy_model,
            policy_optimizer,
            scheduler=CosineAnnealingLR(policy_optimizer, n_updates),
            clip_grad=self.hyperparameters["clip_grad"],
            writer=writer
        )

        return DeepmindAtariBody(
            VAC(features, v, policy, discount_factor=self.hyperparameters["discount_factor"]),
        )

    def test_agent(self):
        features = FeatureNetwork(copy.deepcopy(self.feature_model))
        policy = SoftmaxPolicy(copy.deepcopy(self.policy_model))
        return DeepmindAtariBody(VACTestAgent(features, policy))

vac = preset_builder('vac', default_hyperparameters, VACAtariPreset)
