# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from all.approximation import QDist, FixedTarget
from all.agents import C51
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from .models import nature_c51


def c51(
        # c51-specific hyperparameters
        atoms=51,
        v_min=-10,
        v_max=10,
        # Taken from Extended Data Table 1
        # in https://www.nature.com/articles/nature14236
        # except where noted.
        minibatch_size=32,
        replay_buffer_size=100000, # originally 1e6
        target_update_frequency=1000, # originally 1e4
        discount_factor=0.99,
        action_repeat=4,
        update_frequency=4,
        lr=5e-4, # lr for Adam: Deepmind used RMSprop
        eps=1.5e-4, # stability parameter for Adam
        initial_exploration=1.,
        final_exploration=0.02, # originally 0.1
        final_exploration_frame=1000000,
        replay_start_size=1000,
        device=torch.device('cpu')
):
    # counted by number of updates rather than number of frame
    final_exploration_frame /= action_repeat
    replay_start_size /= action_repeat

    def _c51(env, writer=DummyWriter()):
        model = nature_c51(env, atoms=51).to(device)
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
                    replay_start_size,
                    final_exploration_frame,
                    name="epsilon",
                    writer=writer,
                ),
                discount_factor=discount_factor,
                minibatch_size=minibatch_size,
                replay_start_size=replay_start_size,
                update_frequency=update_frequency,
                writer=writer
            )
        )
    return _c51
