# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch import nn
from torch.optim import Adam
from torch.nn.functional import mse_loss
from all.layers import Flatten
from all.approximation import QNetwork
from all.agents import DQN
from all.policies import GreedyPolicy
from all.utils import ReplayBuffer


def fc_net(env, frames=1):
    return nn.Sequential(
        Flatten(),
        nn.Linear(env.state_space.shape[0] * frames, 256),
        nn.ReLU(),
        nn.Linear(256, env.action_space.n)
    )


def dqn_cc(
        lr=2e-4,
        target_update_frequency=1000,
        annealing_time=10000,
        replay_start_size=32 * 8,
        minibatch_size=64,
        update_frequency=1,
        replay_buffer_size=20000
):
    def _dqn_cc(env):
        model = fc_net(env)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QNetwork(model, optimizer,
                     target_update_frequency=target_update_frequency,
                     loss=mse_loss)
        policy = GreedyPolicy(q, annealing_time=annealing_time)
        replay_buffer = ReplayBuffer(replay_buffer_size)
        return DQN(q, policy, replay_buffer,
                   replay_start_size=replay_start_size,
                   update_frequency=update_frequency,
                   minibatch_size=minibatch_size)
    return _dqn_cc


__all__ = ["dqn_cc"]
