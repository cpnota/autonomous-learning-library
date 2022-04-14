import copy
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import smooth_l1_loss
from all import nn
from all.approximation import QNetwork, FixedTarget
from all.agents import Agent, DQN, DQNTestAgent
from all.bodies import DeepmindAtariBody
from all.logging import DummyLogger
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from all.policies import GreedyPolicy
from all.presets.builder import PresetBuilder
from all.presets.preset import Preset
from all.presets.atari.models import nature_dqn


default_hyperparameters = {
    # Common settings
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
    "initial_exploration": 1.,
    "final_exploration": 0.01,
    "final_exploration_step": 250000,
    "test_exploration": 0.001,
    # Model construction
    "model_constructor": nature_dqn
}


class DQNAtariPreset(Preset):
    """
    Deep Q-Network (DQN) Atari Preset.

    Args:
        env (all.environments.AtariEnvironment): The environment for which to construct the agent.
        name (str): A human-readable name for the preset.
        device (torch.device): The device on which to load the agent.

    Keyword Args:
        discount_factor (float, optional): Discount factor for future rewards.
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
        model_constructor (function): The function used to construct the neural model.
    """

    def __init__(self, env, name, device, **hyperparameters):
        super().__init__(name, device, hyperparameters)
        hyperparameters = {**default_hyperparameters, **hyperparameters}
        self.model = hyperparameters['model_constructor'](env).to(device)
        self.n_actions = env.action_space.n

    def agent(self, logger=DummyLogger(), train_steps=float('inf')):
        n_updates = (train_steps - self.hyperparameters['replay_start_size']) / self.hyperparameters['update_frequency']

        optimizer = Adam(
            self.model.parameters(),
            lr=self.hyperparameters['lr'],
            eps=self.hyperparameters['eps']
        )

        q = QNetwork(
            self.model,
            optimizer,
            scheduler=CosineAnnealingLR(optimizer, n_updates),
            target=FixedTarget(self.hyperparameters['target_update_frequency']),
            logger=logger
        )

        policy = GreedyPolicy(
            q,
            self.n_actions,
            epsilon=LinearScheduler(
                self.hyperparameters['initial_exploration'],
                self.hyperparameters['final_exploration'],
                self.hyperparameters['replay_start_size'],
                self.hyperparameters['final_exploration_step'] - self.hyperparameters['replay_start_size'],
                name="exploration",
                logger=logger
            )
        )

        replay_buffer = ExperienceReplayBuffer(
            self.hyperparameters['replay_buffer_size'],
            device=self.device
        )

        return DeepmindAtariBody(
            DQN(
                q,
                policy,
                replay_buffer,
                discount_factor=self.hyperparameters['discount_factor'],
                loss=smooth_l1_loss,
                minibatch_size=self.hyperparameters['minibatch_size'],
                replay_start_size=self.hyperparameters['replay_start_size'],
                update_frequency=self.hyperparameters['update_frequency'],
            ),
            lazy_frames=True
        )

    def test_agent(self):
        q = QNetwork(copy.deepcopy(self.model))
        policy = GreedyPolicy(
            q,
            self.n_actions,
            epsilon=self.hyperparameters['test_exploration']
        )
        return DeepmindAtariBody(DQNTestAgent(policy))


dqn = PresetBuilder('dqn', default_hyperparameters, DQNAtariPreset)
