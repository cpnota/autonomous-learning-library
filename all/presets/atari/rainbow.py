from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.approximation import QDist, FixedTarget
from all.agents import Rainbow
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.memory import PrioritizedReplayBuffer, NStepReplayBuffer
from all.optim import LinearScheduler
from .models import nature_rainbow


def rainbow(
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
        # Prioritized replay settings
        alpha=0.5,
        beta=0.5,
        # Multi-step learning
        n_steps=3,
        # Distributional RL
        atoms=51,
        v_min=-10,
        v_max=10,
        # Noisy Nets
        sigma=0.5,
        # Model construction
        model_constructor=nature_rainbow
):
    """
    Rainbow Atari Preset.

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
        alpha (float): Amount of prioritization in the prioritized experience replay buffer.
            (0 = no prioritization, 1 = full prioritization)
        beta (float): The strength of the importance sampling correction for prioritized experience replay.
            (0 = no correction, 1 = full correction)
        n_steps (int): The number of steps for n-step Q-learning.
        atoms (int): The number of atoms in the categorical distribution used to represent
            the distributional value function.
        v_min (int): The expected return corresponding to the smallest atom.
        v_max (int): The expected return correspodning to the larget atom.
        sigma (float): Initial noisy network noise.
        model_constructor (function): The function used to construct the neural model.
    """
    def _rainbow(env, writer=DummyWriter()):
        action_repeat = 4
        last_timestep = last_frame / action_repeat
        last_update = (last_timestep - replay_start_size) / update_frequency

        model = model_constructor(env, atoms=atoms, sigma=sigma).to(device)
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

        agent = Rainbow(
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
