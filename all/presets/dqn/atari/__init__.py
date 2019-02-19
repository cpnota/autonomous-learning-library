# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch import nn
from torch.optim import Adam
from all.layers import Flatten, Dueling
from all.approximation import QTabular
from all.agents import DQN
from all.policies import GreedyPolicy
from all.utils import ReplayBuffer

# From the original deep mind paper (https://www.nature.com/articles/nature14236):
#
# "The exact architecture, shown schematically in Fig. 1, is as follows. The input to
# the neural network consists of an 843 843 4 image produced by the preprocessing map w. 
#
# The first hidden layer convolves 32 filters of 8 3 8 with stride 4 with the
# input image and applies a rectifier nonlinearity 31,32. 
#
# The second hidden layer convolves 64 filters of 4 3 4 with stride 2, 
# again followed by a rectifier nonlinearity.
#
# This is followed by a third convolutional layer that convolves 64filters of 3 3 3 with
# stride 1 followed by a rectifier. 
#
# The final hidden layer is fully-connected and consists of 512 rectifier units. 
#
# The output layer is a fully-connected linear layer with a single output for each valid action.
def conv_net(env, frames=4):
    return nn.Sequential(
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(3456, 512),
        nn.ReLU(),
        nn.Linear(512, env.action_space.n)
    )

# "Dueling" architecture modification.
# https://arxiv.org/abs/1511.06581
def dueling_conv_net(env, frames=4):
    return nn.Sequential(
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1),
        nn.ReLU(),
        Flatten(),
        Dueling(
            nn.Sequential(
                nn.Linear(3456, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ),
            nn.Sequential(
                nn.Linear(3456, 256),
                nn.ReLU(),
                nn.Linear(256, env.action_space.n)
            ),
        )
    )

def big_conv_net(env, frames=4):
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
        lr=1e-4,
        target_update_frequency=1000,
        annealing_time=250000,
        initial_epsilon=1.00,
        final_epsilon=0.02,
        buffer_size=200000,
        build_model=dueling_conv_net
        ):
    def _dqn(env):
        model = build_model(env)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QTabular(model, optimizer,
                     target_update_frequency=target_update_frequency)
        policy = GreedyPolicy(q, annealing_time=annealing_time,
                              initial_epsilon=initial_epsilon, final_epsilon=final_epsilon)
        replay_buffer = ReplayBuffer(buffer_size)
        return DQN(q, policy, replay_buffer)
    return _dqn

__all__ = ["dqn", "conv_net", "big_conv_net"]
