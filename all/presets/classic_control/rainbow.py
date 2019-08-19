# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from all.agents import C51
from all.approximation import QDist
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer, NStepReplayBuffer
from all.optim import LinearScheduler
from .models import fc_relu_dist_q


def rainbow(
        device=torch.device("cpu"),
        # vanilla DQN hyperparameters
        discount_factor=0.99,
        final_exploration_frame=10000,
        final_exploration=0.02,
        initial_exploration=1.00,
        lr=1e-4,
        minibatch_size=128,
        replay_buffer_size=20000,
        replay_start_size=1000,
        update_frequency=1,
        # multi-step learning
        n_steps=5,
        # Distributional RL
        atoms=101,
):
    '''
    A (nearly complete) implementation of Rainbow.

    The following enhancements have been applied:
    1. Double Q-learning
    2. Multi-step learning
    3. Distributional RL

    Still to be implemented:
    4. Prioritized Replay
    5. Dueling networks
    6. Noisy Nets
    '''
    def _rainbow(env, writer=DummyWriter()):
        model = fc_relu_dist_q(env, atoms=atoms).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QDist(
            model,
            optimizer,
            env.action_space.n,
            atoms,
            v_min=-100,
            v_max=100,
            writer=writer,
        )
        replay_buffer = NStepReplayBuffer(n_steps, discount_factor, ExperienceReplayBuffer(
            replay_buffer_size,
            device=device
        ))
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
            writer=writer
        )

    return _rainbow


__all__ = ["rainbow"]
