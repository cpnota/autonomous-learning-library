from torch.optim import Adam
from all.agents import C51
from all.approximation import QDist
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from .models import fc_relu_dist_q


def c51(
        # Common settings
        device="cpu",
        discount_factor=0.99,
        # Adam optimizer settings
        lr=1e-4,
        # Training settings
        minibatch_size=128,
        update_frequency=1,
        # Replay buffer settings
        replay_start_size=1000,
        replay_buffer_size=20000,
        # Exploration settings
        initial_exploration=1.00,
        final_exploration=0.02,
        final_exploration_frame=10000,
        # Distributional RL
        atoms=101,
        v_min=-100,
        v_max=100
):
    def _c51(env, writer=DummyWriter()):
        model = fc_relu_dist_q(env, atoms=atoms).to(device)
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
            writer=writer
        )

    return _c51


__all__ = ["c51"]
