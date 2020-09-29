from torch.optim import Adam
from all.agents import Rainbow
from all.approximation import QDist
from all.logging import DummyWriter
from all.memory import (
    PrioritizedReplayBuffer,
    NStepReplayBuffer,
)
from .models import fc_relu_rainbow


def rainbow(
        # Common settings
        device="cpu",
        discount_factor=0.99,
        # Adam optimizer settings
        lr=2e-4,
        # Training settings
        minibatch_size=64,
        update_frequency=1,
        # Replay buffer settings
        replay_buffer_size=20000,
        replay_start_size=1000,
        # Prioritized replay settings
        alpha=0.5,
        beta=0.5,
        # Multi-step learning
        n_steps=5,
        # Distributional RL
        atoms=101,
        v_min=-100,
        v_max=100,
        # Noisy Nets
        sigma=0.5,
        # Model construction
        model_constructor=fc_relu_rainbow
):
    """
    Rainbow classic control preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
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
        model = model_constructor(env, atoms=atoms, sigma=sigma).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QDist(
            model,
            optimizer,
            env.action_space.n,
            atoms,
            v_min=v_min,
            v_max=v_max,
            writer=writer,
        )
        # replay_buffer = ExperienceReplayBuffer(replay_buffer_size, device=device)
        replay_buffer = PrioritizedReplayBuffer(
            replay_buffer_size,
            alpha=alpha,
            beta=beta,
            device=device
        )
        replay_buffer = NStepReplayBuffer(n_steps, discount_factor, replay_buffer)
        return Rainbow(
            q,
            replay_buffer,
            exploration=0.,
            discount_factor=discount_factor ** n_steps,
            minibatch_size=minibatch_size,
            replay_start_size=replay_start_size,
            update_frequency=update_frequency,
            writer=writer,
        )

    return _rainbow


__all__ = ["rainbow"]
