# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch import nn
from torch.optim import Adam
from torch.nn.functional import mse_loss
from all.agents import DQN
from all.approximation import QNetwork
from all.layers import Dueling, Flatten, NoisyLinear
from all.memory import PrioritizedReplayBuffer, ExperienceReplayBuffer
from all.policies import GreedyPolicy

def dueling_fc_net(env, sigma_init):
    return nn.Sequential(
        Flatten(),
        Dueling(
            nn.Sequential(
                nn.Linear(env.state_space.shape[0], 256),
                nn.ReLU(),
                NoisyLinear(256, env.action_space.n, sigma_init=sigma_init)
            ),
            nn.Sequential(
                nn.Linear(env.state_space.shape[0], 256),
                nn.ReLU(),
                NoisyLinear(256, 1, sigma_init=sigma_init)
            )
        )
    )

def rainbow_cc(
        # Vanilla DQN
        minibatch_size=32,
        replay_buffer_size=20000,
        discount_factor=0.99,
        update_frequency=1,
        lr=1e-4,
        replay_start_size=1000,
        build_model=dueling_fc_net,
        # Double Q-Learning
        target_update_frequency=1000,
        # Prioritized Replay
        alpha=0.2,  # priority scaling
        beta=0.6,  # importance sampling adjustment
        final_beta_frame=20000,
        # NoisyNets
        sigma_init=0.1
):
    '''
    Partial implementation of the Rainbow variant of DQN, scaled for classic control environments.

    So far, the enhancements that have been added are:
    1. Double Q-Learning
    2. Prioritized Replay
    3. Dueling Networks
    4. NoisyNets

    Still to be added are:
    5. Multi-step Learning
    6. Distributional RL
    7. Double Q-Learning
    '''
    def _rainbow_cc(env):
        model = build_model(env, sigma_init)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QNetwork(model, optimizer,
                     target_update_frequency=target_update_frequency,
                     loss=mse_loss)
        policy = GreedyPolicy(
            q,
            initial_epsilon=0,
            final_epsilon=0,
            annealing_time=1
        )
        # replay_buffer = ExperienceReplayBuffer(replay_buffer_size)
        replay_buffer = PrioritizedReplayBuffer(
            replay_buffer_size,
            alpha=alpha,
            beta=beta,
            final_beta_frame=final_beta_frame
        )
        return DQN(q, policy, replay_buffer,
                   discount_factor=discount_factor,
                   replay_start_size=replay_start_size,
                   update_frequency=update_frequency,
                   minibatch_size=minibatch_size)
    return _rainbow_cc


__all__ = ["rainbow_cc"]
