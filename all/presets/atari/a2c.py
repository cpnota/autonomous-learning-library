import math
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import A2C, A2CTestAgent
from all.bodies import DeepmindAtariBody
from all.approximation import VNetwork, FeatureNetwork
from all.logging import DummyWriter
from all.policies import SoftmaxPolicy
from .models import nature_features, nature_value_head, nature_policy_head
from ..builder import preset_builder
from ..preset import Preset


default_hyperparameters = {
    # Common settings
    "discount_factor": 0.99,
    # Adam optimizer settings
    "lr": 7e-4,
    "eps": 1.5e-4,
    # Other optimization settings
    "clip_grad": 0.1,
    "entropy_loss_scaling": 0.01,
    "value_loss_scaling": 0.5,
    # Batch settings
    "n_envs": 16,
    "n_steps": 5,
}

class A2CPreset(Preset):
    """
    A2C Atari preset.

    Args:
        discount_factor (float): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        clip_grad (float): The maximum magnitude of the gradient for any given parameter.
            Set to 0 to disable.
        entropy_loss_scaling (float): Coefficient for the entropy term in the total loss.
        value_loss_scaling (float): Coefficient for the value function loss.
        n_envs (int): Number of parallel environments.
        n_steps (int): Length of each rollout.
    """
    def __init__(self, hyperparameters, env, device):
        super().__init__(n_envs=hyperparameters['n_envs'])
        self.value_model = nature_value_head().to(device)
        self.policy_model = nature_policy_head(env).to(device)
        self.feature_model = nature_features().to(device)
        self.hyperparameters = hyperparameters
        self.device = device

    def agent(self, writer=DummyWriter()):
        feature_optimizer = Adam(self.feature_model.parameters(), lr=self.hyperparameters["lr"], eps=self.hyperparameters["eps"])
        value_optimizer = Adam(self.value_model.parameters(), lr=self.hyperparameters["lr"], eps=self.hyperparameters["eps"])
        policy_optimizer = Adam(self.policy_model.parameters(), lr=self.hyperparameters["lr"], eps=self.hyperparameters["eps"])

        features = FeatureNetwork(
            self.feature_model,
            feature_optimizer,
            # scheduler=CosineAnnealingLR(
            #     feature_optimizer,
            #     final_anneal_step,
            # ),
            clip_grad=self.hyperparameters["clip_grad"],
            writer=writer
        )

        v = VNetwork(
            self.value_model,
            value_optimizer,
            # scheduler=CosineAnnealingLR(
            #     value_optimizer,
            #     final_anneal_step,
            # ),
            loss_scaling=self.hyperparameters["value_loss_scaling"],
            clip_grad=self.hyperparameters["clip_grad"],
            writer=writer
        )

        policy = SoftmaxPolicy(
            self.policy_model,
            policy_optimizer,
            # scheduler=CosineAnnealingLR(
            #     policy_optimizer,
            #     final_anneal_step,
            # ),
            clip_grad=self.hyperparameters["clip_grad"],
            writer=writer
        )

        return DeepmindAtariBody(
            A2C(
                features,
                v,
                policy,
                n_envs=self.hyperparameters["n_envs"],
                n_steps=self.hyperparameters["n_steps"],
                discount_factor=self.hyperparameters["discount_factor"],
                entropy_loss_scaling=self.hyperparameters["entropy_loss_scaling"],
                writer=writer
            ),
        )

    def test_agent(self):
        return DeepmindAtariBody(A2CTestAgent(self.feature_model, self.policy_model))

    def n_envs(self):
        return self.hyperparameters['n_envs']

a2c = preset_builder('a2c', default_hyperparameters, A2CPreset)
