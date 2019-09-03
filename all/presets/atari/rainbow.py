# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.approximation import QDist, FixedTarget
from all.agents import C51
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.memory import PrioritizedReplayBuffer, NStepReplayBuffer
from all.optim import LinearScheduler
from .models import nature_rainbow


def rainbow(
        # vanilla DQN parameters
        discount_factor=0.99,
        eps=1.5e-4, # stability parameter for Adam
        lr=6.25e-5,  # requires slightly smaller learning rate than dqn
        minibatch_size=32,
        replay_buffer_size=1000000,
        replay_start_size=100000,
        target_update_frequency=2000,
        update_frequency=4,
        # explicit exploration in addition to noisy nets
        initial_exploration=0.02,
        final_exploration=0.,
        # prioritized replay
        alpha=0.5,
        beta=0.5,
        # multi-step learning
        n_steps=3,
        # Distributional RL
        atoms=51,
        v_min=-10,
        v_max=10,
        # Noisy Nets
        sigma=0.5,
        # Other
        last_frame=40e6,
        device=torch.device('cpu')
):
    '''
    A complete implementation of Rainbow.

    The following enhancements have been applied:
    1. Double Q-learning
    2. Prioritized Replay
    3. Dueling networks
    4. Multi-step learning
    5. Distributional RL
    6. Noisy nets
    '''
    action_repeat = 4
    last_timestep = last_frame / action_repeat
    last_update = (last_timestep - replay_start_size) / update_frequency

    def _rainbow(env, writer=DummyWriter()):
        model = nature_rainbow(env, atoms=atoms, sigma=sigma).to(device)
        optimizer = Adam(model.parameters(), lr=lr, eps=eps)
        q = QDist(
            model,
            optimizer,
            env.action_space.n,
            atoms,
            scheduler=CosineAnnealingLR(optimizer, last_update),
            v_min=v_min,
            v_max=v_max,
            target=FixedTarget(target_update_frequency),
            writer=writer,
        )
        replay_buffer = PrioritizedReplayBuffer(
            replay_buffer_size,
            alpha=alpha,
            beta=beta,
            device=device
        )
        replay_buffer = NStepReplayBuffer(n_steps, discount_factor, replay_buffer)

        agent = C51(
            q,
            replay_buffer,
            exploration=LinearScheduler(
                initial_exploration,
                final_exploration,
                0,
                last_timestep,
                name='exploration',
                writer=writer
            ),
            discount_factor=discount_factor ** n_steps,
            minibatch_size=minibatch_size,
            replay_start_size=replay_start_size,
            update_frequency=update_frequency,
            writer=writer,
        )
        return DeepmindAtariBody(agent, lazy_frames=True, episodic_lives=True)

    return _rainbow
