# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.approximation import QDist, FixedTarget
from all.agents import C51
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from .models import nature_c51


def c51(
        # Common settings
        device=torch.device('cpu'),
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
        initial_exploration=0.02,
        final_exploration=0.,
        # Distributional RL
        atoms=51,
        v_min=-10,
        v_max=10,
):
    # counted by number of updates rather than number of frame
    action_repeat = 4
    last_timestep = last_frame / action_repeat
    last_update = (last_timestep - replay_start_size) / update_frequency

    def _c51(env, writer=DummyWriter()):
        model = nature_c51(env, atoms=atoms).to(device)
        optimizer = Adam(
            model.parameters(),
            lr=lr,
            eps=eps
        )
        q = QDist(
            model,
            optimizer,
            env.action_space.n,
            atoms,
            v_min=v_min,
            v_max=v_max,
            target=FixedTarget(target_update_frequency),
            scheduler=CosineAnnealingLR(optimizer, last_update),
            writer=writer,
        )
        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size,
            device=device
        )
        return DeepmindAtariBody(
            C51(
                q,
                replay_buffer,
                exploration=LinearScheduler(
                    initial_exploration,
                    final_exploration,
                    0,
                    last_timestep,
                    name="epsilon",
                    writer=writer,
                ),
                discount_factor=discount_factor,
                minibatch_size=minibatch_size,
                replay_start_size=replay_start_size,
                update_frequency=update_frequency,
                writer=writer
            ),
            lazy_frames=True
        )
    return _c51
