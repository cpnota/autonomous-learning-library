# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch import nn
from torch.optim import Adam
from all.layers import Flatten
from all.approximation import QTabular
from all.agents import DQN
from all.policies import GreedyPolicy


def fc_net(env, frames=1):
    return nn.Sequential(
        Flatten(),
        nn.Linear(env.state_space.shape[0] * frames, 256),
        nn.ReLU(),
        nn.Linear(256, env.action_space.n)
    )


def dqn_cc(
        lr=1e-4,
        target_update_frequency=1000,
        annealing_time=10000,
        prefetch_size=32 * 8,
        minibatch_size=32,
        update_frequency=1,
):
    def _dqn_cc(env):
        model = fc_net(env)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QTabular(model, optimizer,
                     target_update_frequency=target_update_frequency)
        policy = GreedyPolicy(q, annealing_time=annealing_time)
        return DQN(q, policy,
                   prefetch_size=prefetch_size,
                   update_frequency=update_frequency,
                   minibatch_size=minibatch_size)
    return _dqn_cc


__all__ = ["dqn_cc"]
