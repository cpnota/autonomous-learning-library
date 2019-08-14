# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from all.agents import C51
from all.approximation import QDist
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from .models import fc_relu_dist_q


def c51(
        atoms=51,
        minibatch_size=32,
        replay_buffer_size=20000,
        discount_factor=0.99,
        update_frequency=1,
        lr=1e-4,
        initial_exploration=1.00,
        final_exploration=0.02,
        final_exploration_frame=10000,
        replay_start_size=1000,
        device=torch.device("cpu"),
):
    def _c51(env, writer=DummyWriter()):
        model = fc_relu_dist_q(env, atoms=atoms).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QDist(
            model,
            optimizer,
            env.action_space.n,
            atoms,
            v_min=-30,
            v_max=30,
            writer=writer,
        )
        replay_buffer = ExperienceReplayBuffer(replay_buffer_size, device=device)
        return C51(
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
        )

    return _c51


__all__ = ["c51"]
