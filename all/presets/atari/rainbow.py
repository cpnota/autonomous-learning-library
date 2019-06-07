# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch import nn
from torch.optim import Adam
from torch.nn.functional import smooth_l1_loss
from all.layers import Flatten, Dueling, Linear0
from all.approximation import QNetwork
from all.agents import DQN
from all.bodies import DeepmindAtariBody
from all.experiments import DummyWriter
from all.memory import PrioritizedReplayBuffer
from all.policies import GreedyPolicy

# "Dueling" architecture modification.
# https://arxiv.org/abs/1511.06581


def dueling_conv_net(env, frames=4):
    return nn.Sequential(
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        Flatten(),
        Dueling(
            nn.Sequential(
                nn.Linear(3456, 512),
                nn.ReLU(),
                Linear0(512, 1)
            ),
            nn.Sequential(
                nn.Linear(3456, 512),
                nn.ReLU(),
                Linear0(512, env.action_space.n)
            ),
        )
    )


def rainbow(
        # If None, build defaults
        model=None,
        optimizer=None,
        # vanilla DQN parameters
        minibatch_size=32,
        replay_buffer_size=100000,
        agent_history_length=4,
        target_update_frequency=1000,
        discount_factor=0.99,
        action_repeat=4,
        update_frequency=4,
        lr=5e-4,
        eps=1.5e-4,
        initial_exploration=1.,
        final_exploration=0.02,
        final_exploration_frame=1000000,
        replay_start_size=50000,
        noop_max=30,
        # Prioritized Replay
        alpha=0.5,
        beta=0.4,
        final_beta_frame=200e6,
        device=torch.device('cpu')
):
    '''
    Partial implementation of the Rainbow variant of DQN.

    So far, the enhancements that have been added are:
    1. Prioritized Replay
    2. Dueling Networks

    Still to be added are:
    3. Double Q-Learning
    4. NoisyNets
    5. Multi-step Learning
    6. Distributional RL
    7. Double Q-Learning
    '''
    # counted by number of updates rather than number of frame
    final_exploration_frame /= action_repeat
    replay_start_size /= action_repeat
    final_beta_frame /= action_repeat

    def _rainbow(env, writer=DummyWriter()):
        _model = model
        _optimizer = optimizer
        if _model is None:
            _model = dueling_conv_net(
                env, frames=agent_history_length).to(device)
        if _optimizer is None:
            _optimizer = Adam(
                _model.parameters(),
                lr=lr,
                eps=eps
            )
        q = QNetwork(
            _model,
            _optimizer,
            env.action_space.n,
            target_update_frequency=target_update_frequency,
            loss=smooth_l1_loss,
            writer=writer
        )
        policy = GreedyPolicy(q,
                              env.action_space.n,
                              annealing_start=replay_start_size,
                              annealing_time=final_exploration_frame - replay_start_size,
                              initial_epsilon=initial_exploration,
                              final_epsilon=final_exploration
                              )
        replay_buffer = PrioritizedReplayBuffer(
            replay_buffer_size,
            alpha=alpha,
            beta=beta,
            final_beta_frame=final_beta_frame,
            device=device
        )
        return DeepmindAtariBody(
            DQN(q, policy, replay_buffer,
                discount_factor=discount_factor,
                minibatch_size=minibatch_size,
                replay_start_size=replay_start_size,
                update_frequency=update_frequency,
                ),
            env,
            action_repeat=action_repeat,
            frame_stack=agent_history_length,
            noop_max=noop_max
        )
    return _rainbow


__all__ = ["rainbow", "dueling_conv_net"]
