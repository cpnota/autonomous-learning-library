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
        device="cuda",
        discount_factor=0.99,
        last_frame=40e6,
        # Adam optimizer settings
        lr=1e-4,
        eps=1.5e-4,
        # Training settings
        minibatch_size=32,
        update_frequency=4,
        target_update_frequency=1000,
        # Replay buffer settings
        replay_start_size=80000,
        replay_buffer_size=1000000,
        # Explicit exploration
        initial_exploration=0.02,
        final_exploration=0.,
        # Distributional RL
        atoms=51,
        v_min=-10,
        v_max=10,
        # Model construction
        model_constructor=nature_c51
):
    """
    C51 Atari preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        last_frame (int): Number of frames to train.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        target_update_frequency (int): Number of timesteps between updates the target network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        initial_exploration (int): Initial probability of choosing a random action,
            decayed over course of training.
        final_exploration (int): Final probability of choosing a random action.
        atoms (int): The number of atoms in the categorical distribution used to represent
            the distributional value function.
        v_min (int): The expected return corresponding to the smallest atom.
        v_max (int): The expected return correspodning to the larget atom.
        model_constructor (function): The function used to construct the neural model.
    """
    def _c51(env, writer=DummyWriter()):
        action_repeat = 4
        last_timestep = last_frame / action_repeat
        last_update = (last_timestep - replay_start_size) / update_frequency

        model = model_constructor(env, atoms=atoms).to(device)
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
