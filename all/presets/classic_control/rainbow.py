from torch.optim import Adam
from all.agents import C51
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
):
    """
    A complete implementation of Rainbow.

    The following enhancements have been applied:
    1. Double Q-learning
    2. Prioritized Replay
    3. Dueling networks
    4. Multi-step learning
    5. Distributional RL
    6. Noisy nets
    """

    def _rainbow(env, writer=DummyWriter()):
        model = fc_relu_rainbow(env, atoms=atoms, sigma=sigma).to(device)
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
        return C51(
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
