# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch import nn
from torch.optim import Adam
from all.layers import Flatten
from all.approximation import QTabular
from all.agents import DQN
from all.policies import GreedyPolicy


# def conv_net(env, frames=4):
#     return nn.Sequential(
#         nn.Conv2d(frames, 16, 8, stride=4),
#         nn.ReLU(),
#         nn.Conv2d(16, 32, 4, stride=2),
#         nn.ReLU(),
#         Flatten(),
#         nn.Linear(2816, 256),
#         nn.ReLU(),
#         nn.Linear(256, env.action_space.n)
#     )

def conv_net(env, frames=4):
    return nn.Sequential(
        nn.Conv2d(frames, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(3456, 512),
        nn.ReLU(),
        nn.Linear(512, env.action_space.n)
    )


def dqn(
        lr=1e-5,
        target_update_frequency=1000,
        annealing_time=100000,
        initial_epsilon=1.00,
        final_epsilon=0.02,
        ):
    def _dqn(env):
        model = conv_net(env)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QTabular(model, optimizer,
                     target_update_frequency=target_update_frequency)
        policy = GreedyPolicy(q, annealing_time=annealing_time,
                              initial_epsilon=initial_epsilon, final_epsilon=final_epsilon)
        return DQN(q, policy)
    return _dqn


__all__ = ["dqn"]
