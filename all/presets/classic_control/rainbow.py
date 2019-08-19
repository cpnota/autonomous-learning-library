# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from all.agents import C51
from all.approximation import QDist, PolyakTarget
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer, NStepReplayBuffer
from all.optim import LinearScheduler
from .models import fc_relu_rainbow


def rainbow(
        device=torch.device("cpu"),
        # vanilla DQN hyperparameters
        discount_factor=0.99,
        final_exploration_frame=10000,
        final_exploration=0.02,
        initial_exploration=1.00,
        lr=2e-4,
        minibatch_size=64,
        replay_buffer_size=20000,
        replay_start_size=1000,
        update_frequency=1,
        # multi-step learning
        n_steps=2,
        # Distributional RL
        atoms=101,
        # Noisy Nets
        sigma=0.1,
        # Polyak Target networks
        polyak=0.001
):
    '''
    A (nearly complete) implementation of Rainbow.

    The following enhancements have been applied:
    1. Double Q-learning
    2. Dueling networks
    3. Multi-step learning
    4. Distributional RL
    5. Noisy Nets

    Still to be implemented:
    6. Prioritized Replay
    '''
    def _rainbow(env, writer=DummyWriter()):
        model = fc_relu_rainbow(env, atoms=atoms, sigma=sigma).to(device)
        optimizer = Adam(model.parameters(), lr=lr, eps=1e-3)
        q = QDist(
            model,
            optimizer,
            env.action_space.n,
            atoms,
            v_min=-200,
            v_max=200,
            target=PolyakTarget(polyak),
            writer=writer,
        )
        replay_buffer = NStepReplayBuffer(n_steps, discount_factor, ExperienceReplayBuffer(
            replay_buffer_size,
            device=device
        ))
        return C51(
            q,
            replay_buffer,
            exploration=0,
            discount_factor=discount_factor,
            minibatch_size=minibatch_size,
            replay_start_size=replay_start_size,
            update_frequency=update_frequency,
            writer=writer
        )

    return _rainbow


__all__ = ["rainbow"]
