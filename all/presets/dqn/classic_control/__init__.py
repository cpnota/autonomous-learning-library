# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch import nn
from torch.optim import Adam
from torch.nn.functional import mse_loss
from all.agents import DQN
from all.approximation import QNetwork
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

def dqn_cc(
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
        build_model=fc_net
):
    def _dqn_cc(env):
        model = build_model(env)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QNetwork(model, optimizer,
                     target_update_frequency=target_update_frequency,
                     loss=mse_loss)
        policy = GreedyPolicy(
            q,
            initial_epsilon=initial_exploration,
            final_epsilon=final_exploration,
            annealing_time=final_exploration_frame
        )
        replay_buffer = ExperienceReplayBuffer(replay_buffer_size)
        return DQN(q, policy, replay_buffer,
                   discount_factor=discount_factor,
                   replay_start_size=replay_start_size,
                   update_frequency=update_frequency,
                   minibatch_size=minibatch_size)
    return _dqn_cc


__all__ = ["dqn_cc"]
