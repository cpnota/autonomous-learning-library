import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.approximation import QDist, FixedTarget
from all.agents import C51, C51TestAgent
from all.bodies import DeepmindAtariBody
from all.logging import DummyLogger
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from all.presets.builder import PresetBuilder
from all.presets.preset import Preset
from all.presets.atari.models import nature_c51


default_hyperparameters = {
    "discount_factor": 0.99,
    # Adam optimizer settings
    "lr": 1e-4,
    "eps": 1.5e-4,
    # Training settings
    "minibatch_size": 32,
    "update_frequency": 4,
    "target_update_frequency": 1000,
    # Replay buffer settings
    "replay_start_size": 80000,
    "replay_buffer_size": 1000000,
    # Explicit exploration
    "initial_exploration": 0.02,
    "final_exploration": 0.,
    "final_exploration_step": 250000,
    "test_exploration": 0.001,
    # Distributional RL
    "atoms": 51,
    "v_min": -10,
    "v_max": 10,
    # Model construction
    "model_constructor": nature_c51
}


class C51AtariPreset(Preset):
    """
    Categorical DQN (C51) Atari preset.

    Args:
        env (all.environments.AtariEnvironment): The environment for which to construct the agent.
        name (str): A human-readable name for the preset.
        device (torch.device): The device on which to load the agent.

    Keyword Args:
        discount_factor (float): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        target_update_frequency (int): Number of timesteps between updates the target network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        initial_exploration (float): Initial probability of choosing a random action,
            decayed over course of training.
        final_exploration (float): Final probability of choosing a random action.
        final_exploration_step (int): The step at which exploration decay is finished
        test_exploration (float): The exploration rate of the test Agent
        atoms (int): The number of atoms in the categorical distribution used to represent
            the distributional value function.
        v_min (int): The expected return corresponding to the smallest atom.
        v_max (int): The expected return corresponding to the largest atom.
        model_constructor (function): The function used to construct the neural model.
    """

    def __init__(self, env, name, device, **hyperparameters):
        super().__init__(name, device, hyperparameters)
        self.model = hyperparameters['model_constructor'](env, atoms=hyperparameters['atoms']).to(device)
        self.n_actions = env.action_space.n

    def agent(self, logger=DummyLogger(), train_steps=float('inf')):
        n_updates = (train_steps - self.hyperparameters['replay_start_size']) / self.hyperparameters['update_frequency']

        optimizer = Adam(
            self.model.parameters(),
            lr=self.hyperparameters['lr'],
            eps=self.hyperparameters['eps']
        )

        q = QDist(
            self.model,
            optimizer,
            self.n_actions,
            self.hyperparameters['atoms'],
            v_min=self.hyperparameters['v_min'],
            v_max=self.hyperparameters['v_max'],
            target=FixedTarget(self.hyperparameters['target_update_frequency']),
            scheduler=CosineAnnealingLR(optimizer, n_updates),
            logger=logger,
        )

        replay_buffer = ExperienceReplayBuffer(
            self.hyperparameters['replay_buffer_size'],
            device=self.device
        )

        return DeepmindAtariBody(
            C51(
                q,
                replay_buffer,
                exploration=LinearScheduler(
                    self.hyperparameters['initial_exploration'],
                    self.hyperparameters['final_exploration'],
                    0,
                    self.hyperparameters["final_exploration_step"] - self.hyperparameters["replay_start_size"],
                    name="epsilon",
                    logger=logger,
                ),
                discount_factor=self.hyperparameters["discount_factor"],
                minibatch_size=self.hyperparameters["minibatch_size"],
                replay_start_size=self.hyperparameters["replay_start_size"],
                update_frequency=self.hyperparameters["update_frequency"],
                logger=logger
            ),
            lazy_frames=True,
            episodic_lives=True
        )

    def test_agent(self):
        q_dist = QDist(
            copy.deepcopy(self.model),
            None,
            self.n_actions,
            self.hyperparameters['atoms'],
            v_min=self.hyperparameters['v_min'],
            v_max=self.hyperparameters['v_max'],
        )
        return DeepmindAtariBody(C51TestAgent(q_dist, self.n_actions, self.hyperparameters["test_exploration"]))


c51 = PresetBuilder('c51', default_hyperparameters, C51AtariPreset)
