import copy

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from all.agents import SAC, SACTestAgent
from all.approximation import PolyakTarget, QContinuous
from all.logging import DummyLogger
from all.memory import ExperienceReplayBuffer
from all.policies.soft_deterministic import SoftDeterministicPolicy
from all.presets.builder import PresetBuilder
from all.presets.continuous.models import fc_q, fc_soft_policy
from all.presets.preset import Preset

default_hyperparameters = {
    # Common settings
    "discount_factor": 0.99,
    # Adam optimizer settings
    "lr_q": 1e-3,
    "lr_pi": 1e-3,
    # Training settings
    "minibatch_size": 256,
    "update_frequency": 1,
    "polyak_rate": 0.005,
    # Replay Buffer settings
    "replay_start_size": 5000,
    "replay_buffer_size": 1e6,
    # Exploration settings
    "temperature_initial": 1.0,
    "lr_temperature_scaling": 3e-5,
    "entropy_backups": True,
    "entropy_target_scaling": 1.0,
    # Model construction
    "q1_model_constructor": fc_q,
    "q2_model_constructor": fc_q,
    "policy_model_constructor": fc_soft_policy,
}


class SACContinuousPreset(Preset):
    """
    Soft Actor-Critic (SAC) continuous control preset.

    Args:
        env (all.environments.AtariEnvironment): The environment for which to construct the agent.
        name (str): A human-readable name for the preset.
        device (torch.device): The device on which to load the agent.

    Keyword Args:
        lr_q (float): Learning rate for the Q networks.
        lr_pi (float): Learning rate for the policy network.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        polyak_rate (float): Speed with which to update the target network towards the online network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        temperature_initial (float): Initial value of the temperature parameter.
        lr_temperature (float): Learning rate for the temperature. Should be low compared to other learning rates.
        entropy_target_scaling (float): The target entropy will be -(entropy_target_scaling * env.action_space.shape[0])
        q1_model_constructor (function): The function used to construct the neural q1 model.
        q2_model_constructor (function): The function used to construct the neural q2 model.
        v_model_constructor (function): The function used to construct the neural v model.
        policy_model_constructor (function): The function used to construct the neural policy model.
    """

    def __init__(self, env, name, device, **hyperparameters):
        super().__init__(name, device, hyperparameters)
        self.q1_model = hyperparameters["q1_model_constructor"](env).to(device)
        self.q2_model = hyperparameters["q2_model_constructor"](env).to(device)
        self.policy_model = hyperparameters["policy_model_constructor"](env).to(device)
        self.action_space = env.action_space

    def agent(self, logger=DummyLogger(), train_steps=float("inf")):
        n_updates = (
            train_steps - self.hyperparameters["replay_start_size"]
        ) / self.hyperparameters["update_frequency"]

        q1_optimizer = Adam(self.q1_model.parameters(), lr=self.hyperparameters["lr_q"])
        q1 = QContinuous(
            self.q1_model,
            q1_optimizer,
            scheduler=CosineAnnealingLR(q1_optimizer, n_updates),
            target=PolyakTarget(self.hyperparameters["polyak_rate"]),
            logger=logger,
            name="q1",
        )

        q2_optimizer = Adam(self.q2_model.parameters(), lr=self.hyperparameters["lr_q"])
        q2 = QContinuous(
            self.q2_model,
            q2_optimizer,
            scheduler=CosineAnnealingLR(q2_optimizer, n_updates),
            target=PolyakTarget(self.hyperparameters["polyak_rate"]),
            logger=logger,
            name="q2",
        )

        policy_optimizer = Adam(
            self.policy_model.parameters(), lr=self.hyperparameters["lr_pi"]
        )
        policy = SoftDeterministicPolicy(
            self.policy_model,
            policy_optimizer,
            self.action_space,
            scheduler=CosineAnnealingLR(policy_optimizer, n_updates),
            logger=logger,
        )

        replay_buffer = ExperienceReplayBuffer(
            self.hyperparameters["replay_buffer_size"], device=self.device
        )

        return SAC(
            policy,
            q1,
            q2,
            replay_buffer,
            temperature_initial=self.hyperparameters["temperature_initial"],
            entropy_backups=self.hyperparameters["entropy_backups"],
            entropy_target=(
                -self.action_space.shape[0]
                * self.hyperparameters["entropy_target_scaling"]
            ),
            lr_temperature=self.hyperparameters["lr_temperature_scaling"]
            / self.action_space.shape[0],
            replay_start_size=self.hyperparameters["replay_start_size"],
            discount_factor=self.hyperparameters["discount_factor"],
            update_frequency=self.hyperparameters["update_frequency"],
            minibatch_size=self.hyperparameters["minibatch_size"],
            logger=logger,
        )

    def test_agent(self):
        policy = SoftDeterministicPolicy(
            copy.deepcopy(self.policy_model), space=self.action_space
        )
        return SACTestAgent(policy)


sac = PresetBuilder("sac", default_hyperparameters, SACContinuousPreset)
