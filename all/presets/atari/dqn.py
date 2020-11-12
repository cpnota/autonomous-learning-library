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
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from all.policies import GreedyPolicy
from .models import nature_dqn
from ..builder import preset_builder
from ..preset import Preset


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
    "replay_buffer_size": 500000,
    # Explicit exploration
    "initial_exploration": 1.,
    "final_exploration": 0.01,
    "final_exploration_step": 250000,
}

class DqnPreset(Preset):
    def __init__(self, hyperparameters, env, device='cuda'):
        super().__init__()
        self.model = nature_dqn(env)
        self.hyperparameters = hyperparameters
        self.n_actions = env.action_space.n
        self.device = device

    def agent(self, writer=DummyWriter()):
        optimizer = Adam(
            self.model.parameters(),
            lr=self.hyperparameters['lr'],
            eps=self.hyperparameters['eps']
        )

        q = QNetwork(
            self.model,
            optimizer,
            # scheduler=CosineAnnealingLR(optimizer, last_update),
            target=FixedTarget(self.hyperparameters['target_update_frequency']),
            writer=writer
        )

        policy = GreedyPolicy(
            q,
            self.n_actions,
            epsilon=LinearScheduler(
                self.hyperparameters['initial_exploration'],
                self.hyperparameters['final_exploration'],
                self.hyperparameters['replay_start_size'],
                self.hyperparameters['final_exploration_step'] - self.hyperparameters['replay_start_size'],
                name="epsilon",
                writer=writer
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
        return DeepmindAtariBody(
            DQNTestAgent(copy.deepcopy(self.model), self.n_actions, exploration=0.001)
        )

    def save(self, filename):
        return torch.save(self, filename)

dqn = preset_builder('dqn', default_hyperparameters, DqnPreset)
