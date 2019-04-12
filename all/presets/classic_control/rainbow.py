# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch import nn
from torch.optim import Adam
from torch.nn.functional import mse_loss
from all.agents import DQN
from all.approximation import QNetwork
from all.experiments import DummyWriter
from all.layers import Dueling, Flatten, NoisyLinear
from all.memory import PrioritizedReplayBuffer
from all.policies import GreedyPolicy


def dueling_fc_net(env, sigma_init):
    return nn.Sequential(
        Flatten(),
        Dueling(
            nn.Sequential(
                nn.Linear(env.state_space.shape[0], 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ),
            nn.Sequential(
                nn.Linear(env.state_space.shape[0], 256),
                nn.ReLU(),
                NoisyLinear(256, env.action_space.n, sigma_init=sigma_init)
            )
        )
    )


def rainbow(
        # Vanilla DQN
        minibatch_size=32,
        replay_buffer_size=20000,
        discount_factor=0.99,
        update_frequency=1,
        lr=1e-4,
        replay_start_size=1000,
        build_model=dueling_fc_net,
        # Double Q-Learning
        target_update_frequency=1000,
        # Prioritized Replay
        alpha=0.2,  # priority scaling
        beta=0.6,  # importance sampling adjustment
        final_beta_frame=20000,
        # NoisyNets
        sigma_init=0.1,
        device=torch.device('cpu')
):
    '''
    Partial implementation of the Rainbow variant of DQN, scaled for classic control environments.

    So far, the enhancements that have been added are:
    1. Prioritized Replay
    2. Dueling Networks
    3. NoisyNets

    Still to be added are:
    4. Multi-step Learning
    5. Distributional RL
    6. Double Q-Learning
    '''
    def _rainbow(env, writer=DummyWriter()):
        model = build_model(env, sigma_init).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QNetwork(
            model,
            optimizer,
            env.action_space.n,
            target_update_frequency=target_update_frequency,
            loss=mse_loss,
            writer=writer
        )
        policy = GreedyPolicy(
            q,
            env.action_space.n,
            initial_epsilon=1,
            final_epsilon=0,
            annealing_start=replay_start_size,
            annealing_time=1
        )
        # replay_buffer = ExperienceReplayBuffer(replay_buffer_size)
        replay_buffer = PrioritizedReplayBuffer(
            replay_buffer_size,
            alpha=alpha,
            beta=beta,
            final_beta_frame=final_beta_frame,
            device=device
        )
        return DQN(q, policy, replay_buffer,
                   discount_factor=discount_factor,
                   replay_start_size=replay_start_size,
                   update_frequency=update_frequency,
                   minibatch_size=minibatch_size)
    return _rainbow


__all__ = ["rainbow"]
