# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch import nn
from torch.optim import Adam
from torch.nn.functional import mse_loss
from all.agents import DQN
from all.approximation import QNetwork
from all.layers import Dueling, Flatten
from all.memory import PrioritizedReplayBuffer
from all.policies import GreedyPolicy

def dueling_fc_net(env, frames=1):
    return nn.Sequential(
        Flatten(),
        Dueling(
            nn.Sequential(
                nn.Linear(env.state_space.shape[0] * frames, 256),
                nn.ReLU(),
                nn.Linear(256, env.action_space.n)
            ),
            nn.Sequential(
                nn.Linear(env.state_space.shape[0] * frames, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        )
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
        replay_start_size=32 * 8,
        build_model=dueling_fc_net
):
    '''
    Partial implementation of the Rainbow variant of DQN, scaled for classic control environments.
    
    So far, the enhancements that have been added are:
    1. Dueling architecture
    2. Prioritized experience replay
    '''
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
        replay_buffer = PrioritizedReplayBuffer(replay_buffer_size)
        return DQN(q, policy, replay_buffer,
                   discount_factor=discount_factor,
                   replay_start_size=replay_start_size,
                   update_frequency=update_frequency,
                   minibatch_size=minibatch_size)
    return _dqn_cc

__all__ = ["dqn_cc"]
