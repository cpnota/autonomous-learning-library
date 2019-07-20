# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss
from all.agents import DQN
from all.approximation import QNetwork, FixedTarget
from all.experiments import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.policies import GreedyPolicy
from .models import fc_relu_q

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
        device=torch.device('cpu')
):
    def _dqn(env, writer=DummyWriter()):
        model = fc_relu_q(env).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QNetwork(
            model,
            optimizer,
            env.action_space.n,
            target=FixedTarget(target_update_frequency),
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
