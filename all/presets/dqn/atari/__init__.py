# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch import nn
from torch.optim import RMSprop
from torch.nn.functional import smooth_l1_loss
from all.layers import Flatten, Dueling, Linear0
from all.approximation import QNetwork
from all.agents import DQN
from all.bodies import DeepmindAtariBody
from all.policies import GreedyPolicy
from all.memory import ExperienceReplayBuffer

# Model the original deep mind paper (https://www.nature.com/articles/nature14236):
def conv_net(env, frames=4):
    return nn.Sequential(
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(3456, 512),
        nn.ReLU(),
        Linear0(512, env.action_space.n)
    )

def dqn(
        # If None, build defaults
        model=None,
        optimizer=None,
        # Taken from Extended Data Table 1
        minibatch_size=32,
        replay_buffer_size=250000,  # originally 1e6
        agent_history_length=4,
        target_update_frequency=10000,
        discount_factor=0.99,
        action_repeat=4,
        update_frequency=4,
        lr=0.00025,
        gradient_momentum=0.95,
        squared_gradient_momentum=0.95,
        min_squared_gradient=0.01,
        initial_exploration=1.,
        final_exploration=0.1,
        final_exploration_frame=1000000,
        replay_start_size=50000,
        noop_max=30
):
    # counted by number of updates rather than number of frame
    final_exploration_frame /= action_repeat
    replay_start_size /= action_repeat

    def _dqn(env):
        _model = model
        _optimizer = optimizer
        if _model is None:
            _model = conv_net(env, frames=agent_history_length)
        if _optimizer is None:
            _optimizer = RMSprop(
                _model.parameters(),
                lr=lr,
                momentum=gradient_momentum,
                alpha=squared_gradient_momentum,
                eps=min_squared_gradient
            )
        q = QNetwork(_model, _optimizer,
                     target_update_frequency=target_update_frequency,
                     loss=smooth_l1_loss
                     )
        policy = GreedyPolicy(q,
                              annealing_start=replay_start_size,
                              annealing_time=final_exploration_frame - replay_start_size,
                              initial_epsilon=initial_exploration,
                              final_epsilon=final_exploration
                              )
        replay_buffer = ExperienceReplayBuffer(replay_buffer_size)
        return DeepmindAtariBody(
            DQN(q, policy, replay_buffer,
                discount_factor=discount_factor,
                minibatch_size=minibatch_size,
                replay_start_size=replay_start_size,
                update_frequency=update_frequency,
                ),
            env,
            action_repeat=action_repeat,
            frame_stack=agent_history_length,
            noop_max=noop_max
        )
    return _dqn


__all__ = ["dqn", "conv_net"]
