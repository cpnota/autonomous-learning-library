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
                nn.Linear(3456, 512),
                nn.ReLU(),
                nn.Linear(256, 1)
            ),
            nn.Sequential(
                nn.Linear(3456, 512),
                nn.ReLU(),
                nn.Linear(512, env.action_space.n)
            ),
        )
    )

def dqn(
        minibatch_size=32,
        replay_buffer_size=250000, # originally 1e6
        target_update_frequency=10000,
        discount_factor=0.99,
        update_frequency=4,
        lr=1e-4,
        initial_exploration=1.00,
        final_exploration=0.1,
        final_exploration_frame=250000, # originally 1e6
        replay_start_size=50000,
        build_model=dueling_conv_net
        ):
    def _dqn(env):
        model = build_model(env)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QTabular(model, optimizer,
                     target_update_frequency=target_update_frequency
                    )
        policy = GreedyPolicy(q,
                              annealing_time=final_exploration_frame,
                              initial_epsilon=initial_exploration,
                              final_epsilon=final_exploration
                             )
        replay_buffer = ReplayBuffer(replay_buffer_size)
        return DQN(q, policy, replay_buffer,
                   discount_factor=discount_factor,
                   minibatch_size=minibatch_size,
                   replay_start_size=replay_start_size,
                   update_frequency=update_frequency,
                  )
    return _dqn

__all__ = ["dqn", "conv_net", "dueling_conv_net"]
