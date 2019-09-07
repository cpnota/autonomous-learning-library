# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import smooth_l1_loss
from all.approximation import QNetwork, FixedTarget
from all.agents import DQN
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.policies import GreedyPolicy
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from .models import nature_dqn


def dqn(
        # Common settings
        device=torch.device('cuda'),
        discount_factor=0.99,
        last_frame=40e6,
        # Adam optimizer settings
        lr=1e-4,
        eps=1.5e-4,
        # Training settings
        minibatch_size=32,
        update_frequency=4,
        target_update_frequency=1000,
        # Replay Buffer settings
        replay_start_size=80000,
        replay_buffer_size=1000000,
        # Explicit exploration
        initial_exploration=1.,
        final_exploration=0.01,
        final_exploration_frame=4000000,
):
    action_repeat = 4
    last_timestep = last_frame / action_repeat
    last_update = (last_timestep - replay_start_size) / update_frequency
    final_exploration_step = final_exploration_frame / action_repeat

    def _dqn(env, writer=DummyWriter()):
        model = nature_dqn(env).to(device)
        optimizer = Adam(
            model.parameters(),
            lr=lr,
            eps=eps
        )
        q = QNetwork(
            model,
            optimizer,
            env.action_space.n,
            scheduler=CosineAnnealingLR(optimizer, last_update),
            target=FixedTarget(target_update_frequency),
            loss=smooth_l1_loss,
            writer=writer
        )
        policy = GreedyPolicy(
            q,
            env.action_space.n,
            epsilon=LinearScheduler(
                initial_exploration,
                final_exploration,
                replay_start_size,
                final_exploration_step - replay_start_size,
                name="epsilon",
                writer=writer
            )
        )
        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size,
            device=device
        )
        return DeepmindAtariBody(
            DQN(q, policy, replay_buffer,
                discount_factor=discount_factor,
                minibatch_size=minibatch_size,
                replay_start_size=replay_start_size,
                update_frequency=update_frequency,
                ),
            lazy_frames=True
        )
    return _dqn
