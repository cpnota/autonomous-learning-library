# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch import nn
from torch.optim import Adam
from torch.nn.functional import smooth_l1_loss
from all.layers import Flatten, Dueling
from all.approximation import QNetwork
from all.agents import DQN
from all.bodies import DeepmindAtariBody
from all.policies import GreedyPolicy
from all.memory import PrioritizedReplayBuffer

# "Dueling" architecture modification.
# https://arxiv.org/abs/1511.06581
def dueling_conv_net(env, frames=4):
    return nn.Sequential(
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        Flatten(),
        Dueling(
            nn.Sequential(
                nn.Linear(3456, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            ),
            nn.Sequential(
                nn.Linear(3456, 512),
                nn.ReLU(),
                nn.Linear(512, env.action_space.n)
            ),
        )
    )

def rainbow(
        # Vanilla DQN
        minibatch_size=32,
        replay_buffer_size=250000,  # originally 1e6
        discount_factor=0.99,
        update_frequency=4,
        lr=1e-4,
        replay_start_size=50000,
        build_model=dueling_conv_net,
        # Exploration Schedule
        initial_exploration=1.00,
        final_exploration=0.1,
        final_exploration_frame=250000,  # originally 1e6
        # Double Q-Learning
        target_update_frequency=10000,
        # Prioritized Replay
        alpha=0.6,
        beta=0.4,
        final_beta_frame=100000
):
    '''
    Partial implementation of the Rainbow variant of DQN.

    So far, the enhancements that have been added are:
    1. Double Q-Learning
    2. Prioritized Replay
    3. Dueling Networks

    Still to be added are:
    4. Multi-step Learning
    5. Distributional RL
    6. NoisyNets
    '''
    def _rainbow(env):
        model = build_model(env)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QNetwork(model, optimizer,
                     target_update_frequency=target_update_frequency,
                     loss=smooth_l1_loss)
        policy = GreedyPolicy(
            q,
            initial_epsilon=initial_exploration,
            final_epsilon=final_exploration,
            annealing_time=final_exploration_frame
        )
        # replay_buffer = ExperienceReplayBuffer(replay_buffer_size)
        replay_buffer = PrioritizedReplayBuffer(
            replay_buffer_size,
            alpha=alpha,
            beta=beta,
            final_beta_frame=final_beta_frame
        )
        return DeepmindAtariBody(
            DQN(q, policy, replay_buffer,
                discount_factor=discount_factor,
                minibatch_size=minibatch_size,
                replay_start_size=replay_start_size,
                update_frequency=update_frequency,
                ),
            env
        )
    return _rainbow


__all__ = ["rainbow", "dueling_conv_net"]
