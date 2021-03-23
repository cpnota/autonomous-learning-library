import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import DDPG, DDPGTestAgent
from all.approximation import QContinuous, PolyakTarget
from all.bodies import TimeFeature
from all.logging import DummyWriter
from all.policies import DeterministicPolicy
from all.memory import ExperienceReplayBuffer
from all.presets.builder import PresetBuilder
from all.presets.preset import Preset
from all.presets.continuous.models import fc_q, fc_deterministic_policy


default_hyperparameters = {
    # Common settings
    "discount_factor": 0.98,
    # Adam optimizer settings
    "lr_q": 1e-3,
    "lr_pi": 1e-3,
    # Training settings
    "minibatch_size": 100,
    "update_frequency": 1,
    "polyak_rate": 0.005,
    # Replay Buffer settings
    "replay_start_size": 5000,
    "replay_buffer_size": 1e6,
    # Exploration settings
    "noise": 0.1,
    # Model construction
    "q_model_constructor": fc_q,
    "policy_model_constructor": fc_deterministic_policy
}


class DDPGContinuousPreset(Preset):
    """
    DDPG continuous control preset.

    Args:
        env (all.environments.AtariEnvironment): The environment for which to construct the agent.
        name (str): A human-readable name for the preset.
        device (torch.device): The device on which to load the agent.

    Keyword Args:
        discount_factor (float): Discount factor for future rewards.
        lr_q (float): Learning rate for the Q network.
        lr_pi (float): Learning rate for the policy network.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        polyak_rate (float): Speed with which to update the target network towards the online network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        noise (float): The amount of exploration noise to add.
        q_model_constructor (function): The function used to construct the neural q model.
        policy_model_constructor (function): The function used to construct the neural policy model.
    """

    def __init__(self, env, name, device, **hyperparameters):
        super().__init__(name, device, hyperparameters)
        self.q_model = hyperparameters["q_model_constructor"](env).to(device)
        self.policy_model = hyperparameters["policy_model_constructor"](env).to(device)
        self.action_space = env.action_space

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        n_updates = (train_steps - self.hyperparameters["replay_start_size"]) / self.hyperparameters["update_frequency"]

        q_optimizer = Adam(self.q_model.parameters(), lr=self.hyperparameters["lr_q"])

        q = QContinuous(
            self.q_model,
            q_optimizer,
            target=PolyakTarget(self.hyperparameters["polyak_rate"]),
            scheduler=CosineAnnealingLR(
                q_optimizer,
                n_updates
            ),
            writer=writer
        )

        policy_optimizer = Adam(self.policy_model.parameters(), lr=self.hyperparameters["lr_pi"])
        policy = DeterministicPolicy(
            self.policy_model,
            policy_optimizer,
            self.action_space,
            target=PolyakTarget(self.hyperparameters["polyak_rate"]),
            scheduler=CosineAnnealingLR(
                policy_optimizer,
                n_updates
            ),
            writer=writer
        )

        replay_buffer = ExperienceReplayBuffer(
            self.hyperparameters["replay_buffer_size"],
            device=self.device
        )

        return TimeFeature(DDPG(
            q,
            policy,
            replay_buffer,
            self.action_space,
            noise=self.hyperparameters["noise"],
            replay_start_size=self.hyperparameters["replay_start_size"],
            discount_factor=self.hyperparameters["discount_factor"],
            update_frequency=self.hyperparameters["update_frequency"],
            minibatch_size=self.hyperparameters["minibatch_size"],
        ))

    def test_agent(self):
        policy = DeterministicPolicy(
            copy.deepcopy(self.policy_model),
            None,
            self.action_space,
        )
        return TimeFeature(DDPGTestAgent(policy))


ddpg = PresetBuilder('ddpg', default_hyperparameters, DDPGContinuousPreset)
