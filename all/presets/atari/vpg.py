import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import VPG, VPGTestAgent
from all.approximation import VNetwork, FeatureNetwork
from all.bodies import DeepmindAtariBody
from all.logging import DummyLogger
from all.policies import SoftmaxPolicy
from all.presets.builder import PresetBuilder
from all.presets.preset import Preset
from all.presets.atari.models import nature_features, nature_value_head, nature_policy_head


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
    "min_batch_size": 1000,
    # Model construction
    "feature_model_constructor": nature_features,
    "value_model_constructor": nature_value_head,
    "policy_model_constructor": nature_policy_head
}


class VPGAtariPreset(Preset):
    """
    Vanilla Policy Gradient (VPG) Atari preset.

    Args:
        env (all.environments.AtariEnvironment): The environment for which to construct the agent.
        name (str): A human-readable name for the preset.
        device (torch.device): The device on which to load the agent.

    Keyword Args:
        discount_factor (float): Discount factor for future rewards.
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

    def __init__(self, env, name, device, **hyperparameters):
        super().__init__(name, device, hyperparameters)
        self.value_model = hyperparameters['value_model_constructor']().to(device)
        self.policy_model = hyperparameters['policy_model_constructor'](env).to(device)
        self.feature_model = hyperparameters['feature_model_constructor']().to(device)

    def agent(self, logger=DummyLogger(), train_steps=float('inf')):
        n_updates = train_steps / self.hyperparameters["min_batch_size"]

        feature_optimizer = Adam(self.feature_model.parameters(), lr=self.hyperparameters["lr_pi"], eps=self.hyperparameters["eps"])
        value_optimizer = Adam(self.value_model.parameters(), lr=self.hyperparameters["lr_v"], eps=self.hyperparameters["eps"])
        policy_optimizer = Adam(self.policy_model.parameters(), lr=self.hyperparameters["lr_pi"], eps=self.hyperparameters["eps"])

        features = FeatureNetwork(
            self.feature_model,
            feature_optimizer,
            scheduler=CosineAnnealingLR(feature_optimizer, n_updates),
            clip_grad=self.hyperparameters["clip_grad"],
            logger=logger
        )

        v = VNetwork(
            self.value_model,
            value_optimizer,
            scheduler=CosineAnnealingLR(value_optimizer, n_updates),
            loss_scaling=self.hyperparameters["value_loss_scaling"],
            clip_grad=self.hyperparameters["clip_grad"],
            logger=logger
        )

        policy = SoftmaxPolicy(
            self.policy_model,
            policy_optimizer,
            scheduler=CosineAnnealingLR(policy_optimizer, n_updates),
            clip_grad=self.hyperparameters["clip_grad"],
            logger=logger
        )

        return DeepmindAtariBody(
            VPG(features, v, policy, discount_factor=self.hyperparameters["discount_factor"], min_batch_size=self.hyperparameters["min_batch_size"]),
        )

    def test_agent(self):
        features = FeatureNetwork(copy.deepcopy(self.feature_model))
        policy = SoftmaxPolicy(copy.deepcopy(self.policy_model))
        return DeepmindAtariBody(VPGTestAgent(features, policy))

    def parallel_test_agent(self):
        return self.test_agent()


vpg = PresetBuilder('vpg', default_hyperparameters, VPGAtariPreset)
