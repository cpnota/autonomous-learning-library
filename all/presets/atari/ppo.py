import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import PPO, PPOTestAgent
from all.bodies import DeepmindAtariBody
from all.approximation import VNetwork, FeatureNetwork
from all.logging import DummyWriter
from all.optim import LinearScheduler
from all.policies import SoftmaxPolicy
from all.presets.builder import ParallelPresetBuilder
from all.presets.preset import ParallelPreset
from all.presets.atari.models import nature_features, nature_value_head, nature_policy_head


default_hyperparameters = {
    # Common settings
    "discount_factor": 0.99,
    # Adam optimizer settings
    "lr": 2.5e-4,
    "eps": 1e-5,
    # Other optimization settings
    "clip_grad": 0.5,
    "entropy_loss_scaling": 0.01,
    "value_loss_scaling": 0.5,
    "clip_initial": 0.1,
    "clip_final": 0.01,
    # Batch settings
    "epochs": 4,
    "minibatches": 4,
    "n_envs": 8,
    "n_steps": 128,
    # GAE settings
    "lam": 0.95,
    # Model construction
    "feature_model_constructor": nature_features,
    "value_model_constructor": nature_value_head,
    "policy_model_constructor": nature_policy_head
}


class PPOAtariPreset(ParallelPreset):
    """
    Proximal Policy Optimization (PPO) Atari preset.

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
        entropy_loss_scaling (float): Coefficient for the entropy term in the total loss.
        value_loss_scaling (float): Coefficient for the value function loss.
        clip_initial (float): Value for epsilon in the clipped PPO objective function at the beginning of training.
        clip_final (float): Value for epsilon in the clipped PPO objective function at the end of training.
        epochs (int): Number of times to literature through each batch.
        minibatches (int): The number of minibatches to split each batch into.
        n_envs (int): Number of parallel actors.
        n_steps (int): Length of each rollout.
        lam (float): The Generalized Advantage Estimate (GAE) decay parameter.
        feature_model_constructor (function): The function used to construct the neural feature model.
        value_model_constructor (function): The function used to construct the neural value model.
        policy_model_constructor (function): The function used to construct the neural policy model.
    """

    def __init__(self, env, name, device, **hyperparameters):
        super().__init__(name, device, hyperparameters)
        self.value_model = hyperparameters['value_model_constructor']().to(device)
        self.policy_model = hyperparameters['policy_model_constructor'](env).to(device)
        self.feature_model = hyperparameters['feature_model_constructor']().to(device)

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        n_updates = train_steps * self.hyperparameters['epochs'] * self.hyperparameters['minibatches'] / (self.hyperparameters['n_steps'] * self.hyperparameters['n_envs'])

        feature_optimizer = Adam(self.feature_model.parameters(), lr=self.hyperparameters["lr"], eps=self.hyperparameters["eps"])
        value_optimizer = Adam(self.value_model.parameters(), lr=self.hyperparameters["lr"], eps=self.hyperparameters["eps"])
        policy_optimizer = Adam(self.policy_model.parameters(), lr=self.hyperparameters["lr"], eps=self.hyperparameters["eps"])

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
            PPO(
                features,
                v,
                policy,
                epsilon=LinearScheduler(
                    self.hyperparameters["clip_initial"],
                    self.hyperparameters["clip_final"],
                    0,
                    n_updates,
                    name='clip',
                    writer=writer
                ),
                epochs=self.hyperparameters["epochs"],
                minibatches=self.hyperparameters["minibatches"],
                n_envs=self.hyperparameters["n_envs"],
                n_steps=self.hyperparameters["n_steps"],
                discount_factor=self.hyperparameters["discount_factor"],
                lam=self.hyperparameters["lam"],
                entropy_loss_scaling=self.hyperparameters["entropy_loss_scaling"],
                writer=writer,
            )
        )

    def test_agent(self):
        features = FeatureNetwork(copy.deepcopy(self.feature_model))
        policy = SoftmaxPolicy(copy.deepcopy(self.policy_model))
        return DeepmindAtariBody(PPOTestAgent(features, policy))

    def parallel_test_agent(self):
        return self.test_agent()


ppo = ParallelPresetBuilder('ppo', default_hyperparameters, PPOAtariPreset)
