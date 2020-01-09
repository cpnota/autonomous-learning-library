from torch.optim import Adam
from torch.nn.functional import mse_loss
from all.agents import DDQN
from all.approximation import QNetwork, FixedTarget
from all.logging import DummyWriter
from all.memory import PrioritizedReplayBuffer
from all.optim import LinearScheduler
from all.policies import GreedyPolicy
from .models import dueling_fc_relu_q


def ddqn(
        # Common settings
        device="cpu",
        discount_factor=0.99,
        # Adam optimizer settings
        lr=1e-4,
        # Training settings
        minibatch_size=32,
        update_frequency=1,
        target_update_frequency=1000,
        # Replay buffer settings
        replay_start_size=1000,
        replay_buffer_size=20000,
        # Exploration settings
        initial_exploration=1.00,
        final_exploration=0.02,
        final_exploration_frame=10000,
        # Prioritized replay settings
        alpha=0.2,
        beta=0.6,
):
    '''
    Double Dueling DQN with Prioritized Experience Replay.
    '''
    def _ddqn(env, writer=DummyWriter()):
        model = dueling_fc_relu_q(env).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QNetwork(
            model,
            optimizer,
            env.action_space.n,
            target=FixedTarget(target_update_frequency),
            writer=writer
        )
        policy = GreedyPolicy(
            q,
            env.action_space.n,
            epsilon=LinearScheduler(
                initial_exploration,
                final_exploration,
                replay_start_size,
                final_exploration_frame,
                name="epsilon",
                writer=writer
            )
        )
        replay_buffer = PrioritizedReplayBuffer(
            replay_buffer_size,
            alpha=alpha,
            beta=beta,
            device=device
        )
        return DDQN(q, policy, replay_buffer,
                    loss=mse_loss,
                    discount_factor=discount_factor,
                    replay_start_size=replay_start_size,
                    update_frequency=update_frequency,
                    minibatch_size=minibatch_size)
    return _ddqn


__all__ = ["ddqn"]
