import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import SAC, SACTestAgent
from all.approximation import QContinuous, PolyakTarget, VNetwork
from all.bodies import TimeFeature
from all.logging import DummyWriter
from all.policies.soft_deterministic import SoftDeterministicPolicy
from all.memory import ExperienceReplayBuffer
from all.presets.builder import PresetBuilder
from all.presets.preset import Preset
from all.presets.continuous.models import fc_q, fc_v, fc_soft_policy


default_hyperparameters = {
    # Common settings
    "discount_factor": 0.98,
    # Adam optimizer settings
    "lr_q": 1e-3,
    "lr_v": 1e-3,
    "lr_pi": 1e-4,
    # Training settings
    "minibatch_size": 100,
    "update_frequency": 2,
    "polyak_rate": 0.005,
    # Replay Buffer settings
    "replay_start_size": 5000,
    "replay_buffer_size": 1e6,
    # Exploration settings
    "temperature_initial": 0.1,
    "lr_temperature": 1e-5,
    "entropy_target_scaling": 1.,
    # Model construction
    "q1_model_constructor": fc_q,
    "q2_model_constructor": fc_q,
    "v_model_constructor": fc_v,
    "policy_model_constructor": fc_soft_policy
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
        lr_v (float): Learning rate for the state-value networks.
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
        self.q_1_model = hyperparameters["q1_model_constructor"](env).to(device)
        self.q_2_model = hyperparameters["q2_model_constructor"](env).to(device)
        self.v_model = hyperparameters["v_model_constructor"](env).to(device)
        self.policy_model = hyperparameters["policy_model_constructor"](env).to(device)
        self.action_space = env.action_space

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        n_updates = (train_steps - self.hyperparameters["replay_start_size"]) / self.hyperparameters["update_frequency"]

        q_1_optimizer = Adam(self.q_1_model.parameters(), lr=self.hyperparameters["lr_q"])
        q_1 = QContinuous(
            self.q_1_model,
            q_1_optimizer,
            scheduler=CosineAnnealingLR(
                q_1_optimizer,
                n_updates
            ),
            writer=writer,
            name='q_1'
        )

        q_2_optimizer = Adam(self.q_2_model.parameters(), lr=self.hyperparameters["lr_q"])
        q_2 = QContinuous(
            self.q_2_model,
            q_2_optimizer,
            scheduler=CosineAnnealingLR(
                q_2_optimizer,
                n_updates
            ),
            writer=writer,
            name='q_2'
        )

        v_optimizer = Adam(self.v_model.parameters(), lr=self.hyperparameters["lr_v"])
        v = VNetwork(
            self.v_model,
            v_optimizer,
            scheduler=CosineAnnealingLR(
                v_optimizer,
                n_updates
            ),
            target=PolyakTarget(self.hyperparameters["polyak_rate"]),
            writer=writer,
            name='v',
        )

        policy_optimizer = Adam(self.policy_model.parameters(), lr=self.hyperparameters["lr_pi"])
        policy = SoftDeterministicPolicy(
            self.policy_model,
            policy_optimizer,
            self.action_space,
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

        return TimeFeature(SAC(
            policy,
            q_1,
            q_2,
            v,
            replay_buffer,
            temperature_initial=self.hyperparameters["temperature_initial"],
            entropy_target=(-self.action_space.shape[0] * self.hyperparameters["entropy_target_scaling"]),
            lr_temperature=self.hyperparameters["lr_temperature"],
            replay_start_size=self.hyperparameters["replay_start_size"],
            discount_factor=self.hyperparameters["discount_factor"],
            update_frequency=self.hyperparameters["update_frequency"],
            minibatch_size=self.hyperparameters["minibatch_size"],
            writer=writer
        ))

    def test_agent(self):
        policy = SoftDeterministicPolicy(copy.deepcopy(self.policy_model), space=self.action_space)
        return TimeFeature(SACTestAgent(policy))


sac = PresetBuilder('sac', default_hyperparameters, SACContinuousPreset)
