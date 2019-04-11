# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch import nn
from torch.optim import Adam
from torch.nn.functional import mse_loss
from all.agents import DQN
from all.approximation import QNetwork
from all.experiments import DummyWriter
from all.layers import Flatten
from all.memory import ExperienceReplayBuffer
from all.policies import GreedyPolicy


def fc_net(env, frames=1):
    return nn.Sequential(
        Flatten(),
        nn.Linear(env.state_space.shape[0] * frames, 256),
        nn.ReLU(),
        nn.Linear(256, env.action_space.n)
    )


def dqn(
        minibatch_size=32,
        replay_buffer_size=20000,
        target_update_frequency=1000,
        discount_factor=0.99,
        update_frequency=1,
        lr=1e-4,
        initial_exploration=1.00,
        final_exploration=0.02,
        final_exploration_frame=10000,
        replay_start_size=1000,
        build_model=fc_net,
        device=torch.device('cpu')
):
    def _dqn(env, writer=DummyWriter()):
        model = build_model(env).to(device)
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
            initial_epsilon=initial_exploration,
            final_epsilon=final_exploration,
            annealing_time=final_exploration_frame
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
