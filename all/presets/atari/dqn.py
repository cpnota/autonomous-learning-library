# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from torch.nn.functional import smooth_l1_loss
from all.approximation import QNetwork, FixedTarget
from all.agents import DQN
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.policies import GreedyPolicy
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from .models import nature_dqn


def dqn(
        # Taken from Extended Data Table 1
        # in https://www.nature.com/articles/nature14236
        # except where noted.
        action_repeat=4,
        discount_factor=0.99,
        eps=1.5e-4,
        final_exploration_frame=1000000,
        final_exploration=0.02, # originally 0.1
        initial_exploration=1.,
        lr=2.5e-4,
        minibatch_size=32,
        replay_buffer_size=800000, # originally 1 mil
        replay_start_size=50000,
        target_update_frequency=1000,
        update_frequency=4,
        device=torch.device('cpu')
):
    # counted by number of updates rather than number of frame
    final_exploration_frame /= action_repeat

    def _dqn(env, writer=DummyWriter()):
        _model = nature_dqn(env).to(device)
        _optimizer = Adam(
            _model.parameters(),
            lr=lr,
            eps=eps
        )
        q = QNetwork(
            _model,
            _optimizer,
            env.action_space.n,
            target=FixedTarget(target_update_frequency),
            loss=smooth_l1_loss,
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
            replay_buffer_size,
            device=device
        )
        return DeepmindAtariBody(
            DQN(q, policy, replay_buffer,
                discount_factor=discount_factor,
                minibatch_size=minibatch_size,
                replay_start_size=replay_start_size,
                update_frequency=update_frequency,
                ),
            lazy_frames=True
        )
    return _dqn
