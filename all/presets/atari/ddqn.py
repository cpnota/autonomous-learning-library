# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from torch.nn.functional import smooth_l1_loss
from all.approximation import QNetwork, FixedTarget
from all.agents import DDQN
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.memory import PrioritizedReplayBuffer
from all.optim import LinearScheduler
from all.policies import GreedyPolicy
from .models import nature_ddqn

def ddqn(
        # vanilla DQN parameters
        action_repeat=4,
        discount_factor=0.99,
        eps=1.5e-4,
        final_exploration_frame=1000000,
        final_exploration=0.02,
        initial_exploration=1.,
        lr=1e-4,
        minibatch_size=32,
        replay_buffer_size=100000,
        replay_start_size=50000,
        target_update_frequency=1000,
        update_frequency=4,
        # Prioritized Replay
        alpha=0.4,
        beta=0.6,
        device=torch.device('cpu')
):
    '''
    Double Dueling DQN with Prioritized Experience Replay.
    '''
    # counted by number of updates rather than number of frame
    final_exploration_frame /= action_repeat

    def _ddqn(env, writer=DummyWriter()):
        _model = nature_ddqn(env, frames=action_repeat).to(device)
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
        replay_buffer = PrioritizedReplayBuffer(
            replay_buffer_size,
            alpha=alpha,
            beta=beta,
            device=device
        )
        return DeepmindAtariBody(
            DDQN(q, policy, replay_buffer,
                 discount_factor=discount_factor,
                 minibatch_size=minibatch_size,
                 replay_start_size=replay_start_size,
                 update_frequency=update_frequency,
                ),
            lazy_frames=True
        )
    return _ddqn
