# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from all.agents import DQN
from all.approximation import QNetwork, FixedTarget
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from all.policies import GreedyPolicy
from .models import fc_relu_q

def dqn(
        # Common settings
        device=torch.device('cpu'),
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
        initial_exploration=1.,
        final_exploration=0.01,
        final_exploration_frame=4000000,
):
    def _dqn(env, writer=DummyWriter()):
        model = fc_relu_q(env).to(device)
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
        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size, device=device)
        return DQN(q, policy, replay_buffer,
                   discount_factor=discount_factor,
                   replay_start_size=replay_start_size,
                   update_frequency=update_frequency,
                   minibatch_size=minibatch_size)
    return _dqn


__all__ = ["dqn"]
